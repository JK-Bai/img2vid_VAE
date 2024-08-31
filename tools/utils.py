import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 gateconv3d_bak
class gateconv3d_bak(nn.Module):
    def __init__(self, innum, outnum, kernel, stride, pad):
        super(gateconv3d_bak, self).__init__()
        self.conv = nn.Conv3d(innum, outnum * 2, kernel, stride, pad, bias=True)
        self.bn = nn.BatchNorm3d(outnum * 2)

    def forward(self, x):
        return F.glu(self.bn(self.conv(x)), 1) + x

# 定义 gateconv3d
class gateconv3d(nn.Module):
    def __init__(self, innum, outnum, kernel, stride, pad):
        super(gateconv3d, self).__init__()
        self.conv = nn.Conv3d(innum, outnum, kernel, stride, pad, bias=True)
        self.bn = nn.BatchNorm3d(outnum)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), 0.2)

# 定义 convblock
class convblock(nn.Module):
    def __init__(self, innum, outnum, kernel, stride, pad):
        super(convblock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(innum, outnum, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(outnum),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.main(x)

# 定义 convbase
class convbase(nn.Module):
    def __init__(self, innum, outnum, kernel, stride, pad):
        super(convbase, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(innum, outnum, kernel, stride, pad),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.main(x)

# 定义 upconv
class upconv(nn.Module):
    def __init__(self, innum, outnum, kernel, stride, pad):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(innum, outnum * 2, kernel, stride, pad),
            nn.BatchNorm2d(outnum * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outnum * 2, outnum, kernel, stride, pad),
            nn.BatchNorm2d(outnum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def forward(self, x):
        return self.main(x)

# 定义 getflow
class getflow(nn.Module):
    def __init__(self):
        super(getflow, self).__init__()
        self.main = nn.Sequential(
            upconv(64, 16, 5, 1, 2),
            nn.Conv2d(16, 2, 5, 1, 2),
        )

    def forward(self, x):
        return self.main(x)

# 定义 get_occlusion_mask
class get_occlusion_mask(nn.Module):
    def __init__(self):
        super(get_occlusion_mask, self).__init__()
        self.main = nn.Sequential(
            upconv(64, 16, 5, 1, 2),
            nn.Conv2d(16, 2, 5, 1, 2),
        )

    def forward(self, x):
        return F.sigmoid(self.main(x))


# 定义 get_frames
class get_frames(nn.Module):
    def __init__(self, opt):
        super(get_frames, self).__init__()
        self.main = nn.Sequential(
            upconv(64, 16, 5, 1, 2),
            nn.Conv2d(16, opt.input_channel, 5, 1, 2)
        )

    def forward(self, x):
        return torch.sigmoid(self.main(x))
