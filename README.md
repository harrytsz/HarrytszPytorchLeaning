# HarrytszPytorchLeaning
The Process of Learning Pytorch

Created by Harrytsz at 2019/3/28/ in The WeskLake of HangZhou.

## Torch 
> 2002年发布 Torch
> 2011年发布 Torch7

## PyTorch
> 2016年10月发布0.1， THNN
> 2018年12月发布1.0，Caffe2
> Facebook， AI Research

## 大浪淘沙

|Theano|Torch7|Caffe|
|:----:|:----:|:---:|
|TensorFlow|PyTorch+THNN|Caffe2|
|TensorFlow eager|PyTorch+Caffe2|PyTorch1.0|

## 动态图

## 静态图
$\cdot$ 自建命名体系
$\cdot$ 自建时序控制
$\cdot$ 难以介入

TensorFlow2.0 => 动态图优先！！！

## 深度学习库能做什么？
$\cdot$ GPU加速
$\cdot$ 自动求导
$\cdot$ 常用网络层API

## 自动求导

$$y = a^{2} \times x + b \times x + c$$

$$\begin{Bmatrix}
\frac{\partial y}{\partial a} = 2a \times x = 2ax
\\ \frac{\partial y}{\partial b} = x
\\ \frac{\partial y}{\partial c} = 1
\end{Bmatrix}$$


## 常用 API

| Tensor 运算 | 神经网络 |
|:---------:|:-------:|
| Torch.add | Nn.Linear |
| Torch.mul | Nn.ReLU |
| Torch.matmul | Nn.Conv2d |
| Torch.view | Nn.Softmax |
| Torch.expand | Nn.Sigmoid |
| Torch.cat | Nn.CrossEntropyLoss |
|....|....|
