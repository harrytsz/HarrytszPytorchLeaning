## GPU 加速的案例

import torch
import time
print(torch.__version__)           # 显示 Pytorch 的版本
print(torch.cuda.is_available())   # 测试 cuda 是否可用

a = torch.randn(10000, 1000)       # 创建 矩阵 10000 X 1000
b = torch.randn(1000, 2000)        # 创建 矩阵 1000 X 2000

t0 = time.time()                   # 记录程序开始时刻
c = torch.matmul(a, b)             # CPU模式的矩阵乘法
t1 = time.time()                   # 记录程序结束时刻
print(a.device, t1-t0, c.norm(2))  # 打印CPU模式的矩阵乘法执行时间

device = torch.device('cuda')
a = a.to(device)                   # 将 A、B 矩阵搬到 CUDA 上
b = b.to(device)

t0 = time.time()
c = torch.matmul(a, b)             # 使用 GPU 模式计算矩阵乘法
t2 = time.time()
print(a.device, t2-t0, c.norm(2))

t0 = time.time()                   # 重复执行一次代码，保证运行时间记录的准确性
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2-t0, c.norm(2))


######################### Test 执行结果：

1.0.1
True
cpu 0.40003013610839844 tensor(141324.2031)
cuda:0 0.540036678314209 tensor(141324.2031, device='cuda:0')
cuda:0 0.01599907875061035 tensor(141324.2031, device='cuda:0')
