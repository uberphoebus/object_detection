import numpy as np

# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0

# model prediction
def forward(x):
    return w * x

# loss MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

# gradient : dJ/dw = 1/N 2x(w * x - y)
def grad(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()

print(f'prediction before: f(5) = {forward(5)}')

# train
learning_rate = 0.01
n_iters = 10
for epoch in range(n_iters):
    # pred : forward
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # grad
    dw = grad(X, Y, y_pred)
    
    # update weights
    w -= learning_rate * dw
    
    if epoch % 1 == 0:
        print(f'epoch {epoch + 1}: w = {w:.5f}, loss = {l:.5f}')

print(f'prediction after: f(5) = {forward(5)}')
