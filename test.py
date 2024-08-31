import os
from models.Vae import VAE
from tools import utils
from config.opts import parse_opts
from tools import utils_tools
from data.VedioDataset import VideoDataset
from torch.utils.data import DataLoader
import torch

opt = parse_opts()

class flowgen_test(object):

    def __init__(self, opt):
        self.opt = opt
        self.workspace = os.path.dirname(os.path.realpath(__file__))  # 当前文件所在目录
        self.modeldir = os.path.join(self.workspace)  # 模型保存路径
        self.sampledir = os.path.join(self.workspace, 'sample', '')

        test_path = 'D:/ImageToVedio/ucf/IceDancing'
        test_dataset = VideoDataset(video_dir=test_path, num_frame=30, input_size=(128, 128), returnpath=False)
        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

    def load_model(self, model_path):
        vae = VAE(hallucination=False, opt=self.opt).cuda()
        checkpoint = torch.load(model_path)
        vae.load_state_dict(checkpoint['vae'])
        vae.eval()  # 设置模型为评估模式
        return vae

    def test(self):
        vae = self.load_model(os.path.join('D:\PycharmProjects\img2vid_VAE\models\checkpoints\icedancing_model.pth'))

        with torch.no_grad():
            for i,sample in enumerate(self.test_loader):
                data = sample.cuda()
                frame1 = data[:, 0, :, :, :]
                noise_bg = torch.randn(frame1.size()).cuda()

                y_pred_before_refine, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw = vae(frame1, data, noise_bg)

                # 保存结果
                utils_tools.save_samples(data, y_pred_before_refine, y_pred, flow, mask_fw, mask_bw, i,
                                         self.sampledir, self.opt, eval=True, useMask=True)

                print(f"Processed batch {i + 1}/{len(self.test_loader)}")

if __name__ == '__main__':
    a = flowgen_test(opt)
    a.test()
