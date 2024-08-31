import torch.optim
import torch.optim as optim
import os, time
from models.Vae import *
from tools import losses
from config.opts import parse_opts
from tools import utils_tools
from data.VedioDataset import VideoDataset
from torch.utils.data import DataLoader
from tools import save_videos
import torch
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

opt = parse_opts()

class flowgen(object):

    def __init__(self, opt):

        self.opt = opt
        self.workspace = os.path.dirname(os.path.realpath(__file__))  # 当前文件所在目录
        self.modeldir = os.path.join(self.workspace, 'models')  # 模型保存路径
        self.sampledir = os.path.join(self.workspace, 'sample', '')
        self.parameterdir = os.path.join(self.workspace, 'params')  # 参数保存路径
        self.useHallucination = False

        if not os.path.exists(self.parameterdir):
            os.makedirs(self.parameterdir)
        train_path = 'D:/PycharmProjects/img2vid_VAE/output/train'
        test_path = 'D:/PycharmProjects/img2vid_VAE/output/test'

        train_dataset = VideoDataset(video_dir=train_path, num_frame=30, input_size=(128, 128), returnpath=False)
        test_dataset = VideoDataset(video_dir=test_path, num_frame=30, input_size=(128, 128), returnpath=False)

        self.train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
        self.test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=1)

    def train(self):

        opt = self.opt

        iteration = 0
        rank = 0

        vae = VAE(hallucination=False, opt=self.opt).cuda()
        checkpoint_path = os.path.join(os.getcwd(), 'models', 'checkpoints', 'playingviolin_model.pth')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), weights_only=False)
        model_dict = vae.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['vae'].items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        vae.load_state_dict(model_dict)

        # 初始化学习率
        learning_rate_matched = 0.00001  # 能正常加载的学习率
        learning_rate_unmatched = 0.0001  # 不能正常加载的学习率
        default_learning_rate = 0.00005  # 默认学习率

        # 标记哪些权重是匹配的，哪些是不匹配的
        matched_weights = []
        unmatched_weights = []
        all_param_names = set([name for name, _ in vae.named_parameters()])

        for name, param in vae.named_parameters():
            if name in pretrained_dict:
                matched_weights.append(name)
            else:
                unmatched_weights.append(name)

        other_weights = all_param_names - set(matched_weights) - set(unmatched_weights)

        # 创建优化器，并为不同类别的参数设置不同的学习率
        optimizer = torch.optim.Adam([
            {'params': [param for name, param in vae.named_parameters() if name in matched_weights],
             'lr': learning_rate_matched},
            {'params': [param for name, param in vae.named_parameters() if name in unmatched_weights],
             'lr': learning_rate_unmatched},
            {'params': [param for name, param in vae.named_parameters() if name in other_weights],
             'lr': default_learning_rate}
        ])
        # 定义学习率调度器
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
        vae.train()

        objective_func = losses.losses_multigpu_only_mask(opt, vae.floww)

        torch.backends.cudnn.benchmark = True

        for epoch in  range(opt.num_epochs):

            print('Epoch {}/{}'.format(epoch, opt.num_epochs - 1))
            print('-' * 10)
            global_loss = 0
            for sample in iter(self.train_loader):

                data = sample.cuda()
                frame1 = data[:, 0, :, :, :]
                frame2 = data[:, 1:, :, :, :]
                print(frame1.shape)
                print(data.shape)
                noise_bg = torch.randn(frame1.size()).cuda()

                start = time.time()

                vae.train()

                optimizer.zero_grad()

                y_pred_before_refine, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature = vae(
                    frame1, data, noise_bg)

                flowloss, reconloss, reconloss_back, reconloss_before, kldloss, flowcon, sim_loss, vgg_loss, mask_loss = objective_func(
                    frame1, frame2,
                    y_pred, mu, logvar, flow, flowback,
                    mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature,
                    y_pred_before_refine=y_pred_before_refine)

                global_loss = (flowloss + 2. * reconloss + reconloss_back + reconloss_before + kldloss * self.opt.lamda + flowcon + sim_loss + vgg_loss + 0.1 * mask_loss)

                global_loss.backward()

                optimizer.step()
                end = time.time()

                if iteration % 20 == 0 and rank == 0:
                    print(
                        "iter {} (epoch {}), recon_loss = {:.6f}, recon_loss_back = {:.3f}, "
                        "recon_loss_before = {:.3f}, flow_loss = {:.6f}, flow_consist = {:.3f}, kl_loss = {:.6f}, "
                        "img_sim_loss= {:.3f}, vgg_loss= {:.3f}, mask_loss={:.3f}, time/batch = {:.3f}"
                        .format(iteration, epoch, reconloss.item(), reconloss_back.item(), reconloss_before.item(),
                                flowloss.item(), flowcon.item(),
                                kldloss.item(), sim_loss.item(), vgg_loss.item(), mask_loss.item(), end - start))

                if iteration % 500 == 0:
                    # 为每个 iteration 创建一个子目录
                    suffix = "_500"
                    iteration_dir = os.path.join(self.sampledir, f'{iteration:06d}{suffix}')
                    # 如果目录不存在，则创建它
                    if not os.path.exists(iteration_dir):
                        os.makedirs(iteration_dir)
                    utils_tools.save_samples(data, y_pred_before_refine, y_pred, flow, mask_fw, mask_bw, iteration,
                                             iteration_dir, opt)

                if iteration % 2000 == 0:
                    with torch.no_grad():
                        vae.eval()
                        val_sample = next(iter(self.test_loader))
                        data = val_sample.cuda()
                        frame1 = data[:, 0, :, :, :]
                        noise_bg = torch.randn(frame1.size()).cuda()
                        y_pred_before_refine, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw = vae(frame1, data, noise_bg)

                        suffix = "_2000"
                        iteration_dir = os.path.join(self.sampledir, f'{iteration:06d}{suffix}')
                        if not os.path.exists(iteration_dir):
                            os.makedirs(iteration_dir)
                        utils_tools.save_samples(data, y_pred_before_refine, y_pred, flow, mask_fw, mask_bw, iteration,
                                             iteration_dir, opt,
                                             eval=True, useMask=True)
                iteration += 1

            if epoch % 100 == 0:
                checkpoint_dir = os.path.join(os.getcwd(), 'models', 'checkpoints')
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                suffix = "_output"
                epoch_dir = os.path.join(checkpoint_dir, '{:06d}{}'.format(epoch, suffix))
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                checkpoint_path = os.path.join(epoch_dir, 'checkpoint_epoch_{:06d}.pth'.format(epoch))
                print("model saved to {}".format(checkpoint_path))
                torch.save({'vae': vae.state_dict(), 'optimizer': optimizer.state_dict()},
                           checkpoint_path)
            scheduler.step(global_loss)


if __name__ == '__main__':
    a = flowgen(opt)
    a.train()