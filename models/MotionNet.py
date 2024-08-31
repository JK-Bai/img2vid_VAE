import torch
import torch.nn as nn


class MotionNet(nn.Module):
    def __init__(self, opt, input_channel=90, output_channel=int(1024 / 2)):  # 输入通道调整为90
        super(MotionNet, self).__init__()

        # 定义主卷积模块
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, 2, 1, bias=False),  # 输出尺寸 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 输出尺寸 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # 输出尺寸 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 输出尺寸 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),  # 输出尺寸 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False)  # 调整后的输出尺寸 8x8，通道数调整为256
        )

        # 重新计算展平后的特征向量维度
        self.fc_input_dim = 256 * 8 * 8  # 如果输出特征图为256通道，尺寸为8x8

        self.fc1 = nn.Linear(self.fc_input_dim, 1024)  # 调整全连接层输入维度
        self.fc2 = nn.Linear(self.fc_input_dim, 1024)

    def forward(self, x):
        # 卷积输出展平为向量
        temp = self.main(x).view(-1, self.fc_input_dim)

        # 计算均值和对数方差
        mu = self.fc1(temp)
        logvar = self.fc2(temp)

        return mu, logvar

