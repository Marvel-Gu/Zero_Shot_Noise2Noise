import torch
import torchvision
from utils import load_img, add_noise

def add_noise(x,noise_level):

    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level/255, x.shape)
        noisy = torch.clamp(noisy,0,1)

    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x)/noise_level

    return noisy


noise_type = 'gauss' # Either 'gauss' or 'poiss'
device = torch.device('cpu')
origin = load_img("pictures/ORIGIN.PNG").to(device)
img1= add_noise(origin, 25)  # add_noise定义在utils包中
trans = torchvision.transforms.ToPILImage()
pic1=trans(img1)
pic1.save("pictures/NOISY.png")

she