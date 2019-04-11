import torch
from pprint import pprint

x = torch.ones(2, 2, requires_grad=True)

pprint(x)

y = x*x + 1

print(y)

pprint(y.grad_fn)

y.backward(torch.tensor([[1., 1.], [1., 2.]]))

print(x.grad)