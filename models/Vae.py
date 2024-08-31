import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg19, VGG19_Weights
from models.MotionNet import MotionNet
from models.Encoder import Encoder
from models.Decoder import Decoder
from tools.utils import convbase,getflow,get_frames,get_occlusion_mask
from tools import ops
from .Vgg_utils import my_vgg
from torch.autograd import Variable as Vb

mean = torch.FloatTensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1])
std = torch.FloatTensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1])

class VAE(nn.Module):
    def __init__(self, hallucination=False, opt=None, refine=True, motion_dim=512):
        super(VAE, self).__init__()

        self.opt = opt
        self.hallucination = hallucination

        # 统一的运动网络
        self.motion_net = MotionNet(opt,int(opt.num_frames * opt.input_channel), motion_dim)

        self.encoder = Encoder(opt)
        self.flow_decoder = Decoder(opt)

        if self.hallucination:
            self.raw_decoder = Decoder(opt)
            self.predict = get_frames(opt)

        self.zconv = convbase(256 + 16, int(16*self.opt.num_predicted_frames), 3, 1, 1)

        self.floww = ops.flowwrapper()
        self.fc = nn.Linear(1024, 1024)
        self.flownext = getflow()  # 用于生成前向光流
        self.flowprev = getflow()  # 用于生成后向光流
        self.get_mask = get_occlusion_mask()

        self.refine = refine
        if self.refine:
            from models.Vgg_128 import RefineNet
            self.refine_net = RefineNet(num_channels=opt.input_channel)
        # 使用默认权重加载
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT)
        self.vgg_net = my_vgg(vgg19_model)
        for param in self.vgg_net.parameters():
            param.requires_grad = False

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Vb(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return Vb(mu.data.new(mu.size()).normal_())

    def _normalize(self, x):
        gpu_id = x.get_device()
        return (x - mean.cuda(gpu_id)) / std.cuda(gpu_id)

    def forward(self, x, data, noise_bg, z_m=None):
        # 修改后的 frame 处理
        frame1 = data[:, 0, :, :, :]
        frame2 = data[:, 1:, :, :, :]

        opt = self.opt

        # y 的处理需要考虑到新的输入帧数 (30)
        y = torch.cat(
            [frame1, frame2.contiguous().view(-1, opt.num_predicted_frames * opt.input_channel, opt.input_size[1],
                                              opt.input_size[0]) -
             frame1.repeat(1, opt.num_predicted_frames, 1, 1)], 1)

        # 编码器网络
        enco1, enco2, enco3,  codex = self.encoder(x)

        # 调整 motion_net 的输入，考虑到新的帧数
        mu, logvar = self.motion_net(
            y.contiguous().view(-1, opt.num_frames * opt.input_channel, opt.input_size[1], opt.input_size[0]))

        if z_m is None:
            z_m = self.reparameterize(mu, logvar)
        codey = self.zconv(
            torch.cat([self.fc(z_m).view(-1, 16, 8, 8), codex], 1))
        codex = torch.unsqueeze(codex, 2).repeat(1, 1, opt.num_predicted_frames, 1, 1)  # bs,256,4,8,8
        codey = torch.cat(torch.chunk(codey.unsqueeze(2), opt.num_predicted_frames, 1), 2)  # bs,16,4,8,8
        z = torch.cat(torch.unbind(torch.cat([codex, codey], 1), 2), 0)  # (256L, 272L, 8L, 8L)   272-256=16

        # 流解码器网络
        flow_deco4 = self.flow_decoder(enco1, enco2, enco3, z)
        flow = torch.cat(self.flownext(flow_deco4).unsqueeze(2).chunk(opt.num_predicted_frames, 0), 2)
        flowback = torch.cat(self.flowprev(flow_deco4).unsqueeze(2).chunk(opt.num_predicted_frames, 0), 2)

        masks = torch.cat(self.get_mask(flow_deco4).unsqueeze(2).chunk(opt.num_predicted_frames, 0),
                          2)  # (64, 2, 4, 128, 128)
        mask_fw = masks[:, 0, ...]
        mask_bw = masks[:, 1, ...]

        '''Use mask before warpping'''
        output = ops.warp(x, flow, opt, self.floww, mask_fw)

        y_pred = output

        # 进一步优化
        if self.refine:
            y_pred = ops.refine(output, flow, mask_fw, self.refine_net, opt, noise_bg)

        if self.training:
            tmp1 = output.contiguous().view(-1, opt.input_channel, opt.input_size[0], opt.input_size[1])
            tmp2 = frame2.contiguous().view(-1, opt.input_channel, opt.input_size[0], opt.input_size[1])

            if opt.input_channel == 1:
                tmp1 = tmp1.repeat(1, 3, 1, 1)
                tmp2 = tmp2.repeat(1, 3, 1, 1)

            prediction_vgg_feature = self.vgg_net(self._normalize(tmp1))
            gt_vgg_feature = self.vgg_net(self._normalize(tmp2))

            return output, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature

        else:
            return output, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw

