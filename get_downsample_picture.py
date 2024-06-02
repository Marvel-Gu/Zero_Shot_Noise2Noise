import torch
import torchvision
from downsample import downsample
from utils import load_img

device = torch.device('cpu')
img = load_img('pictures/NOISY.PNG').to(device)
img1, img2 = downsample(img)

# 假设img1和img2的批次大小为1，我们通过索引[0]来获取单个图像
img1 = img1[0]
img2 = img2[0]

trans = torchvision.transforms.ToPILImage()
pic1 = trans(img1)
pic2 = trans(img2)

# 将PIL图像保存为文件
pic1.save("noisy1.png")
pic2.save("noisy2.png")

