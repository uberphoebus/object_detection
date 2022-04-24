# 1. design model(input, output size, forward pass)
# 2. construct loss and optim
# 3. training loop : forward, backward, update weights

import torch
import torch.nn as nn

# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], {8}], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

# model prediction
class LinerRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinerRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.lin(x)
        return x

model = LinerRegression(input_size, output_size)

print(f'prediction before: f(5) = {model(X_test)}')

# train
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # pred : forward
    y_pred = model(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # grad : backward
    l.backward() # dl/dw
    
    # update weights
    optimizer.step()
    
    # zero grads
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.5f}, loss = {l:.5f}')

print(f'prediction after: f(5) = {model(X_test)}')