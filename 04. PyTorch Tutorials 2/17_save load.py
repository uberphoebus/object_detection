import torch
import torch.nn as nn


"""
PATH = './'
model = nn.Linear()

#### complete model ####
torch.save(model, PATH)

# model class must be defined
model = torch.load(PATH)
model.eval()

#### state dict ####
torch.save(model.state_dict(), PATH)

# model must be created again with params
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH)) # take loaded dict
model.eval()
"""

############################################################

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# save model
model = Model(n_input_features=6)

# .pth pytorch filename
FILE = r'C:\workspace\pythonProject\object_detection\04. PyTorch Tutorials 2\model1.pth'
torch.save(model, FILE)


# load model (lazy)
model = torch.load(FILE)
model.eval()

for param in model.parameters():
    print(param)


# load model (prefered)
model = Model(n_input_features=6)
FILE = r'C:\workspace\pythonProject\object_detection\04. PyTorch Tutorials 2\model2.pth'
torch.save(model.state_dict(), FILE)

loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)


# train
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model.state_dict())
print(optimizer.state_dict())

checkpoint = {
    'epoch': 90,
    'model_state': model.state_dict(), 
    'optim_state': optimizer.state_dict()
}
CH_PATH = r'C:\workspace\pythonProject\object_detection\04. PyTorch Tutorials 2\checkpoint.pth'
torch.save(checkpoint, CH_PATH)

loaded_checkpoint = torch.load(CH_PATH)
epoch = loaded_checkpoint['epoch']
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(checkpoint['model_state']) # if using gpu, add 'map_location="cuda:0"'
optimizer.load_state_dict(checkpoint['optim_state'])

print(optimizer.state_dict())