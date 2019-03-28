# 自动求导
import torch
from torch import autograd

x = torch.tensor(1.0)                         # 新建变量 x
a = torch.tensor(1.0, requires_grad = True)   # 新建变量 a ，并声明需要对 a 求导
b = torch.tensor(2.0, requires_grad = True)   # 新建变量 b ，并声明需要对 b 求导
c = torch.tensor(3.0, requires_grad = True)   # 新建变量 c ，并声明需要对 c 求导

y = a**2*x + b*x + c

print('before:', a.grad, b.grad, c.grad)
grads = autograd.grad(y, [a, b, c])
print('after:', grads[0], grads[1], grads[2])


################## Test 测试结果：

#before: None None None
#after: tensor(2.) tensor(1.) tensor(1.)