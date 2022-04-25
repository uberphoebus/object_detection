import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0. prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1. model : f = wx + b, sigmoid
class Logistic(nn.Module):
    def __init__(self, n_input_features):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Logistic(n_features)


# 2. loss and optim
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 3. train loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # backward
    loss.backward()
    
    # update
    optimizer.step()
    
    # empty grads
    optimizer.zero_grad()
    
    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch + 1}, loss = {loss.item():.5f}')


# test
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round() # sigmoid
    
    # metrics
    acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'acc = {acc:.4f}')