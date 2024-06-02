### 尝试对Zero-Shot-Noise2Noise进行复现

根据CVPR2023论文Zero-Shot-Noise2Noise复现了一个去噪模型,[原论文地址](https://arxiv.org/abs/2303.11253)

#### 结果

**原图和加噪后的图像**对比
<div class="half" style="text-align: center;">
<img src="./pictures/ORIGIN.png" width=400 alt="原图"/>
<img src="./pictures/NOISY.png" width=400 alt="加噪图像"/>
</div>

加噪后图像的**下取样对**:
<div class="half" style="text-align: center;">
<img src="./noisy1.png" width=400/>
<img src="./noisy2.png" width=400/>
</div>

使用0.001和0.002的学习率分别训练2000轮, 损失函数随训练轮数的变化曲线如下:

<div class="half" style="text-align: center;">
<img src="./myplot.png" width=400 alt="lr:0.001"/>
<img src="./myplot1.png" width=400 alt="lr:0.002"/>
</div>

最终去噪后的结果对比:

<div class="half" style="text-align: center;">
<img src="./pictures/ORIGIN.png" width=400 alt="原图"/>
<img src="./pictures/NOISY.png" width=400 alt="加噪图像"/>
</div>

<div class="half" style="text-align: center;">
<img src="./pictures/result.jpg" width=400 alt="lr:0.001"/>
<img src="./pictures/result1.jpg" width=400 alt="lr:0.002"/>
</div>




