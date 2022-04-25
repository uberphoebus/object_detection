import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

lr = 0.1
model = nn.Linear(10, 1)

############################################################

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lambda1 = lambda epoch: epoch / 10
scheduler = lr_scheduler.LambdaLR(optimizer, lambda1)

print(optimizer.state_dict())

for epoch in range(5): # increasing lr (lambda1)
    optimizer.step()
    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])

############################################################

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lambda2 = lambda epoch: 0.95
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda2)

print(optimizer.state_dict())

for epoch in range(5):
    optimizer.step()
    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])

############################################################