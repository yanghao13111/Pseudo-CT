import torch
import torch.nn as nn
from torchsummary import summary

# Residual Block
class Res(nn.Module):
    def __init__(self, filters):
        super(Res, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),  # 先卷積
            nn.BatchNorm2d(filters),  # 再正規化
            nn.LeakyReLU(0.01),  # 最後激活
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.01)
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
        self.final_block = nn.Sequential(
            nn.Conv2d(filters, 1, kernel_size=1, bias=False),  # 1 channel for CT
            nn.Tanh()  # 使用 Tanh 作為輸出激活函數
        )

    def forward(self, img_in):
        img_out = self.final_block(img_in)
        return img_out

# 預訓練 UNet
def Pretrain():
    # 從 torch.hub 中加載預訓練模型
    model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5, verbose=False)
    
    # 修改輸入層
    in_filters = model.inc.double_conv[3].out_channels
    model.inc = Init(in_filters)  # 自定義的 Init 輸入層

    # 修改輸出層
    out_filters = model.outc.conv.in_channels
    model.outc = Final(out_filters)  # 自定義的 Final 輸出層

    return model

# 主函數
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nTraining on: {device}\n')

    # 加載預訓練模型並修改輸入輸出層
    model = Pretrain().to(device)
    
    # 顯示模型結構
    print(summary(model, input_size=(7, 192, 192), batch_size=2))
