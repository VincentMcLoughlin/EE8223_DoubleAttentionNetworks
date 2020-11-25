import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
bs = 128
num_epochs = 20
learning_rate = 0.001
learning_rate = 0.00001/2
numChannels = 3

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../cifar-data-data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../cifar-data-data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=bs, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=bs, 
                                          shuffle=False)

initialOutputSize = 56
block1Size = 32
convBlock2Size = 64
convBlock3Size = 128
convBlock4Size = 256
convBlock5Size = 512
numClasses = 10

class InitialConvblock2(nn.Module):
  def __init__(self, in_size, out_size): #In size of 224, outsize of 56
    super().__init__()
    self.kernelSize = 5
    self.stride = 2
    self.pad = 2
    self.layer = nn.Sequential(
        nn.Conv2d(in_size, out_size, self.kernelSize, self.stride, self.pad),
        nn.BatchNorm2d(out_size),
        nn.ReLU(inplace=True),
        nn.AdaptiveMaxPool2d(initialOutputSize)
    )
    self.conv1 = nn.Conv2d(in_size, out_size, self.kernelSize, self.stride, self.pad)

  def InitialConvblock2(self, x):
    x = self.layer(x)
    return x

  def forward(self, x): return self.InitialConvblock2(x)

class ResBlock(nn.Module):
  def __init__(self, in_size, hidden_layer_size, out_size, cutSize=False):
    super().__init__()
    self.kernelSize1 = 1
    self.kernelSize3 = 3
    self.stride1 = 1
    self.stride2 = 2
    if cutSize:
      self.initialStride=2
    else:
      self.initialStride=1
    self.pad0 = 0
    self.pad1 = 1
    self.conv1 = nn.Conv2d(in_size, hidden_layer_size, self.kernelSize1, self.initialStride, self.pad0)
    self.conv2 = nn.Conv2d(hidden_layer_size, hidden_layer_size, self.kernelSize3, self.stride1, self.pad0)
    self.conv3 = nn.Conv2d(hidden_layer_size, out_size, self.kernelSize1, self.stride1, self.pad1)

    self.convShortcut = nn.Conv2d(in_size, out_size, self.kernelSize1, self.stride1, self.pad0, bias=False)
    self.convShortcutResize = nn.Conv2d(in_size, out_size, self.kernelSize1, self.stride2, self.pad0, bias=False) #Use if Final shortcut for block
    
    self.batchnorm1 = nn.BatchNorm2d(hidden_layer_size, affine=True)
    self.batchnorm2 = nn.BatchNorm2d(out_size, affine=True)
    self.batchnorm3 = nn.BatchNorm2d(out_size, affine=True)
    self.batchnormShortcut = nn.BatchNorm2d(out_size, affine=True)

    self.relu = nn.ReLU(inplace=True)

  def resblock(self, x, resizeShortcut=False):
    if resizeShortcut:
      shortcut = self.batchnormShortcut(self.convShortcutResize(x))      
    else:
      shortcut = self.batchnormShortcut(self.convShortcut(x))
    
    blockOutput = self.relu(self.batchnorm3(self.conv3(self.relu(self.batchnorm1(self.conv2(self.relu(self.batchnorm1(self.conv1(x)))))))))
    
    return shortcut + blockOutput

  def forward(self, x, resizeShortcut=False): return self.resblock(x, resizeShortcut)


class ResNet(nn.Module):

  def __init__(self, n_classes=10):
    super().__init__()  

    self.nClasses = n_classes
    #self.convBlock1 = InitialConvblock(numChannels, block1Size)
    self.convBlock2 = InitialConvblock2(numChannels, block1Size)
  
    self.block2a = ResBlock(block1Size, convBlock2Size, convBlock2Size*4) #Should be input of 32, next 64, next 64 and out 256
    self.block2b = ResBlock(convBlock2Size*4, convBlock2Size,convBlock2Size*4) #should be input of 256, next layer is 64, and out 256
    
    self.block3a = ResBlock(convBlock2Size*4, convBlock3Size, convBlock3Size*4, True)
    self.block3b = ResBlock(convBlock3Size*4, convBlock3Size, convBlock3Size*4)


    self.block4a = ResBlock(convBlock3Size*4, convBlock4Size, convBlock4Size*4, True)
    self.block4b = ResBlock(convBlock4Size*4, convBlock4Size, convBlock4Size*4)

    self.block5a = ResBlock(convBlock4Size*4, convBlock5Size, convBlock5Size*4, True)
    self.block5b = ResBlock(convBlock5Size*4, convBlock5Size, convBlock5Size*4)

    self.globalAvgPool = nn.AdaptiveAvgPool2d(1) 
    self.linearLayer = nn.Linear(convBlock5Size*4, self.nClasses) #num classes
    self.relu = nn.ReLU(inplace=True)


  def forward(self, x):
    x = self.convBlock2(x)
    
    x = self.block2b(self.block2b(self.block2a(x)))

    x = self.block3b(self.block3b(self.block3b(self.block3a(x, True))))

    x = self.block4b(self.block4b(self.block4b(self.block4b(self.block4b(self.block4a(x, True))))))

    x = self.block5b(self.block5b(self.block5a(x, True)))

    x = self.globalAvgPool(x)
    x = x.view(-1, convBlock5Size*4)
    x = self.relu(self.linearLayer(x))

    return x

#model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
model = ResNet(numClasses).to(device)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Epoch Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')