import torch
import time
print(torch.__version__)
print(torch.cuda.is_available())

a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1-t0, c.norm(2))

device = torch.device('cuda')
a = a.to(device)
b = b.to(device)

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2-t0, c.norm(2))

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2-t0, c.norm(2))


######################### Test

1.0.1
True
cpu 0.40003013610839844 tensor(141324.2031)
cuda:0 0.540036678314209 tensor(141324.2031, device='cuda:0')
cuda:0 0.01599907875061035 tensor(141324.2031, device='cuda:0')
