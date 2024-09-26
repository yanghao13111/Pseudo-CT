import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

def get_mse(predicts, labels):
    return F.mse_loss(predicts, labels)

def gradient_difference_loss(predicts, labels):
    def compute_gradients(image):
        # 計算圖像的水平方向和垂直方向的梯度
        dx = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
        dy = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
        return dx, dy

    pred_dx, pred_dy = compute_gradients(predicts)
    label_dx, label_dy = compute_gradients(labels)

    gdl_loss = torch.mean(torch.abs(pred_dx - label_dx)) + torch.mean(torch.abs(pred_dy - label_dy))
    return gdl_loss

def get_ssim(predicts, labels):
    # 使用 SSIM 衡量預測和真實標籤之間的相似度
    ssim_loss = 1 - ssim(predicts, labels, data_range=1, size_average=True)
    return ssim_loss

def combined_loss(predicts, labels, alpha=0.5, beta=0.3, gamma=0.2):
    mse_loss = get_mse(predicts, labels)
    gdl_loss = gradient_difference_loss(predicts, labels)
    ssim_loss = get_ssim(predicts, labels)

    return alpha * mse_loss + beta * gdl_loss + gamma * ssim_loss

if __name__ == '__main__':
    torch.manual_seed(0)
    inputs = torch.rand((3, 1, 192, 192)).float()  # 模擬 MRI 預測結果
    torch.manual_seed(1)
    target = torch.rand((3, 1, 192, 192)).float()  # 模擬真實的 CT 結果

    # 計算 MSE 損失
    mse_loss = get_mse(inputs, target)
    print(f'MSE Loss: {mse_loss.item()}')

    # 計算 GDL 損失
    gdl_loss = gradient_difference_loss(inputs, target)
    print(f'GDL Loss: {gdl_loss.item()}')

    # 計算 SSIM 損失
    ssim_loss = get_ssim(inputs, target)
    print(f'SSIM Loss: {ssim_loss.item()}')

    # 計算加權組合損失 (MSE + GDL + SSIM)
    combined = combined_loss(inputs, target, alpha=0.5, beta=0.3, gamma=0.2)
    print(f"Combined Loss (MSE + GDL + SSIM): {combined.item()}")
