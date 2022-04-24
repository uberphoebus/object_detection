import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0. prepare data
X_np, y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
y = y.view(y.shape[0], 1) # reshape

n_samples, n_features = X.shape


# 1. model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2. loss and optim
learning_rate = 0.01
creiterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3. train loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward
    y_pred = model(X)
    loss = creiterion(y_pred, y)
    
    # backward
    loss.backward()
    
    # update
    optimizer.step()
    
    # empty grads
    optimizer.zero_grad()
    
    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch + 1}: loss = {loss.item():.5f}')

# plot
pred = model(X).detach()
plt.plot(X_np, y_np, 'ro')
plt.plot(X_np, pred, 'b')
plt.show()