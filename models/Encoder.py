import torch.nn as nn
from tools.utils import convbase, convblock

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.econv1 = convbase(opt.input_channel, 32, 4, 2, 1)  # 32,64,64
        self.econv2 = convblock(32, 64, 4, 2, 1)  # 64,32,32
        self.econv3 = convblock(64, 128, 4, 2, 1)  # 128,16,16
        self.econv4 = convblock(128, 256, 4, 2, 1)  # 256,8,8

    def forward(self, x):
        enco1 = self.econv1(x)  # 32
        enco2 = self.econv2(enco1)  # 64
        enco3 = self.econv3(enco2)  # 128
        codex = self.econv4(enco3)  # 256
        return enco1, enco2, enco3, codex
