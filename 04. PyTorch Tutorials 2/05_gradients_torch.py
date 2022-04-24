import torch

# f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

# loss MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

# gradient : dJ/dw = 1/N 2x(w * x - y)

print(f'prediction before: f(5) = {forward(5)}')

# train
learning_rate = 0.01
n_iters = 100
for epoch in range(n_iters):
    # pred : forward
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # grad : backward
    l.backward() # dl/dw
    
    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # zero grads
    w.grad.zero_()
    
    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}: w = {w:.5f}, loss = {l:.5f}')

print(f'prediction after: f(5) = {forward(5)}')
