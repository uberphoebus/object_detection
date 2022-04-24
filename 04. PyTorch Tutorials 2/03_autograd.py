import torch

x = torch.randn(3, requires_grad=True)
print(x)

# prevent tracking grad

x.requires_grad_(False)
print(x)
x.requires_grad_(True)

y = x.detach()
print(y)
x.requires_grad_(True)

with torch.no_grad():
    y = x + 2
    print(y)


# empty grads

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights * 3).sum()
    model_output.backward()
    
    print(weights.grad)
    
    # empty grads
    weights.grad.zero_()