import torch
import torch.nn as nn
from torch.nn import functional as F

from downsample import downsample


class Tnet(nn.Module):
    def __init__(self):
        super().__init__()
        # 调整卷积层的填充参数以保持尺寸一致
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        X1 = F.relu(self.conv1(X))
        X2 = F.relu(self.conv2(X1))
        X3 = self.conv3(X2)

        # 检查每个卷积层后的张量尺寸
        print(f"Input size: {X.size()}")
        print(f"After conv1: {X1.size()}")
        print(f"After conv2: {X2.size()}")
        print(f"After conv3: {X3.size()}")

        return X + X3


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src_down1, src_down2, dst, down1_dst, down2_dst):
        loss_res = 0.5 * ((down1_dst - src_down2) ** 2 + (down2_dst - src_down1) ** 2)
        dst_down1, dst_down2 = downsample(dst)
        loss_cons = 0.5 * ((down1_dst - dst_down1) ** 2 + (down2_dst - dst_down2) ** 2)
        return (loss_res + loss_cons).mean()



