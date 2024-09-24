import torch
import torch.nn as nn
from torchsummary import summary


# Residual Block
class Res(nn.Module):
    def __init__(self, filters):
        super(Res, self).__init__()
        self.res_block = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.01),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.01),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, img_in):
        img_out = self.res_block(img_in)
        return img_in + img_out  # 跳躍連接


# 初始化區塊
class Init(nn.Module):
    def __init__(self, filters):
        super(Init, self).__init__()
        self.conv = nn.Conv2d(7, filters, kernel_size=3, padding=1, bias=False)  # 7 channels for MRI
        self.drop = nn.Dropout2d(0.2)
        self.res = Res(filters)

    def forward(self, img_in):
        img_out = self.conv(img_in)
        img_out = self.drop(img_out)
        img_out = self.res(img_out)
        return img_out


# 最終輸出區塊
class Final(nn.Module):
    def __init__(self, filters):
        super(Final, self).__init__()
        self.final_block = nn.Conv2d(filters, 1, kernel_size=1, bias=False)  # 1 channel for CT

    def forward(self, img_in):
        img_out = self.final_block(img_in)
        return img_out


# UNet 模型架構
class UNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=1, filters=64):
        super(UNet, self).__init__()
        self.init = Init(filters)
        self.down1 = nn.Conv2d(filters, filters * 2, kernel_size=3, stride=2, padding=1)
        self.res1 = Res(filters * 2)
        self.down2 = nn.Conv2d(filters * 2, filters * 4, kernel_size=3, stride=2, padding=1)
        self.res2 = Res(filters * 4)

        self.up1 = nn.ConvTranspose2d(filters * 4, filters * 2, kernel_size=2, stride=2)
        self.res3 = Res(filters * 2)
        self.up2 = nn.ConvTranspose2d(filters * 2, filters, kernel_size=2, stride=2)
        self.final = Final(filters)

    def forward(self, img_in):
        x1 = self.init(img_in)
        x2 = self.res1(self.down1(x1))
        x3 = self.res2(self.down2(x2))

        x = self.up1(x3) + x2  # 跳躍連接
        x = self.res3(x)
        x = self.up2(x) + x1  # 跳躍連接

        img_out = self.final(x)
        return img_out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    print(summary(model, input_size=(7, 192, 192), batch_size=2))
