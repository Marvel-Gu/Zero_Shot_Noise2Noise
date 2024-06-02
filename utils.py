import torchvision
import numpy as np
import time
import torch.nn as nn
import torch
import random
from PIL import Image


noise_type = 'gauss' # Either 'gauss' or 'poiss'
noise_level = 25     # Pixel range is 0-255 for Gaussian, and 0-1 for Poission

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def net_init(net):
    for param in net.parameters():
        if type(param) == nn.Conv2d:
            nn.init.xavier_uniform_(param)


def load_img(image:str):
    trans = torchvision.transforms.ToTensor()
    img = Image.open(image)
    return trans(img)



def add_noise(x,noise_level):

    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level/255, x.shape)
        noisy = torch.clamp(noisy,0,1)

    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x)/noise_level

    return noisy




# 计时器模块
class Timer(object):
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()
    
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times) / len(self.times)
    
    def sum(self):
        return sum(self.times)

    # 累计执行时间
    def cumsum(self):
        return np.array(self.times).cumsum().tolist

