import torch.nn as nn
from tools.utils import convbase, convblock,upconv,gateconv3d
import torch


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        self.dconv1 = convblock(256 + 16, 256, 3, 1, 1)  # 256,8,8
        self.dconv2 = upconv(256, 128, 3, 1, 1)  # 128,16,16
        self.dconv3 = upconv(256, 64, 3, 1, 1)  # 64,32,32
        self.dconv4 = upconv(128, 32, 3, 1, 1)  # 32,64,64
        self.gateconv1 = gateconv3d(64, 64, 3, 1, 1)
        self.gateconv2 = gateconv3d(32, 32, 3, 1, 1)

    def forward(self, enco1, enco2, enco3, z):
        opt = self.opt
        deco1 = self.dconv1(z)  # .view(-1,256,4,4,4)# bs*4,256,8,8
        deco2 = torch.cat(torch.chunk(self.dconv2(deco1).unsqueeze(2), opt.num_predicted_frames, 0), 2)  # bs*4,128,16,16
        deco2 = torch.cat(torch.unbind(torch.cat([deco2, torch.unsqueeze(enco3, 2).repeat(1, 1, opt.num_predicted_frames, 1, 1)], 1), 2), 0)
        deco3 = torch.cat(self.dconv3(deco2).unsqueeze(2).chunk(opt.num_predicted_frames, 0), 2)  # 128,32,32
        deco3 = self.gateconv1(deco3)
        deco3 = torch.cat(torch.unbind(torch.cat([deco3, torch.unsqueeze(enco2, 2).repeat(1, 1, opt.num_predicted_frames, 1, 1)], 1), 2), 0)
        deco4 = torch.cat(self.dconv4(deco3).unsqueeze(2).chunk(opt.num_predicted_frames, 0), 2)  # 32,4,64,64
        deco4 = self.gateconv2(deco4)
        deco4 = torch.cat(torch.unbind(torch.cat([deco4, torch.unsqueeze(enco1, 2).repeat(1, 1, opt.num_predicted_frames, 1, 1)], 1), 2), 0)
        return deco4
