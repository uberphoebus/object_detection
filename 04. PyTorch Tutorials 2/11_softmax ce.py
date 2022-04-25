import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(f'numpy outputs : {outputs}')

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(f'torch outputs : {outputs}')

######################################################################

def cross_entropy(actual, pred):
    loss = -np.sum(actual * np.log(pred))
    return loss # / float(pred.shape[0])

y = np.array([1, 0, 0]) # y must be one hot encoded

# y_pred probas
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(y, y_pred_good)
l2 = cross_entropy(y, y_pred_bad)
print(f'loss1 numpy : {l1}')
print(f'loss2 numpy : {l2}')


loss = nn.CrossEntropyLoss()

y = torch.tensor([2, 0, 1]) # 3 samples
# nsamples x nclasses = 3 x 3
y_pred_good = torch.tensor([[0.2, 1.0, 2.1], [2.0, 1.0, 0.1], [0.3, 3.0, 0.1]])
y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)
print(f'loss1 torch : {l1.item()}')
print(f'loss2 torch : {l2.item()}')

_, preds1 = torch.max(y_pred_good, 1)
_, preds2 = torch.max(y_pred_bad, 1)
print(preds1, preds2) # [0], [1]