import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sys
import matplotlib.pyplot as plt


###################################################
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/mnist')
writer = SummaryWriter('runs/mnist2') # another writer (change lr)
# how to launch : tensorboard --logdir=runs
###################################################


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 # 28 x 28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
# learning_rate = 0.001
learning_rate = 0.01


# dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                           transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                          transform=transforms.ToTensor())

# dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, shuffle=False)

# check batch
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape) # batch_size, channels, height, width

# plot
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()


###################################################
# show image by tensorboard
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)
###################################################

# model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out) # no softmax for multi classification
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


###################################################
# plot graph by tensorboard
writer.add_graph(model, samples.reshape(-1, 28*28))
# writer.close() # outputs flushed
# sys.exit() # stop pipeline
###################################################


# train loop
n_total_steps = len(train_loader)

running_loss = 0.0
running_correct = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28 -> 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        # add for tensorboard
        running_loss += loss.item()
        
        _, pred = torch.max(outputs.data, 1)
        running_correct += (pred == labels).sum().item()
        
        if (i + 1) % 100 == 0:
            print(f'epch {epoch+1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
            ###################################################
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0
            ###################################################

# test

# for tensorboard
class_labels, class_preds = [], []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        # value, index
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        # for tensorboard
        class_probas_batch = [F.softmax(output, dim=0) for output in outputs]
        class_preds.append(class_probas_batch)
        class_labels.append(predicted)
    
    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels) # make 1d tensor
    
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')
    
    
    ###################################################
    classes = range(10)
    for i in classes:
        labels_i = class_labels == i
        preds_i = class_preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
    ###################################################