import torch
from torch.nn import functional as F
from utils import load_img


def downsample(img):
    # 确保img是四维张量
    if img.dim() == 3:
        img = img.unsqueeze(0)  # 添加一个批次维度

    # 现在img应该是四维的，我们可以安全地解包形状
    B, C, H, W = img.shape

    # 创建滤波器，每个组一个滤波器，共2个滤波器
    filter1 = torch.FloatTensor([[0, 0.5], [0.5, 0]]).to(img.device)
    filter2 = torch.FloatTensor([[0.5, 0], [0, 0.5]]).to(img.device)

    # 扩展滤波器以匹配通道数
    filter1 = filter1.repeat(C, 1, 1, 1)
    filter2 = filter2.repeat(C, 1, 1, 1)

    # 应用滤波器，groups参数应该等于通道数C
    output1 = F.conv2d(img, filter1, stride=2, groups=C)
    output2 = F.conv2d(img, filter2, stride=2, groups=C)

    return output1, output2

def shape_test(tensor):
    res = downsample(tensor)
    return res[0].shape, res[1].shape


if __name__ == "__main__":
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    X = torch.ones((1, 3, 64, 48)).to(device)
    print(shape_test(X))
    img = load_img('pictures/NOISY.PNG').to(device)
    print(img)
    down_img1, down_img2 = downsample(img)
    print(down_img1.shape)




