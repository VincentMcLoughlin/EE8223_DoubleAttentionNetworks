import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import math
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class IdentityShortcut(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityShortcut, self).__init__()        
        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1
    blockType = "basic"
    def __init__(self, in_size, out_size, stride=1): #May need resizing parameter
        super().__init__()

        self.expansion=1
        self.blockType = "basic"

        if in_size != out_size:
            self.projection = IdentityShortcut(out_size, in_size, stride)
        else:
            self.projection = None
        
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_size)
        self.relu1 = nn.ReLU(inplace=True) 
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_size) 
        self.relu2 = nn.ReLU(inplace=True)        

    def forward(self, x):

        res = x       
        output = self.conv1(x)
        output = self.batchnorm1(output)
        output = self.relu1(output)        
        output = self.conv2(output)
        output = self.batchnorm2(output)

        if self.projection:
            res = self.projection(x)

        output += res
        output = self.relu2(output)

        return output

class BottleneckBlock(nn.Module):

    blockType = "bottleneck"
    def __init__(self, in_size, out_size, stride=1):
        super().__init__()
        self.expansion = 4        

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=1, bias=False) #Should bias be true?
        self.batchnorm1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, bias=False, padding=1, stride=stride) #Do we need padding?
        self.batchnorm2 = nn.BatchNorm2d(out_size)
        #Relu again

        self.conv3 = nn.Conv2d(out_size, out_size*4, kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_size*4)
        #Relu Again

        if stride != 1 or in_size != out_size*4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size*4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size*4)
            )
        else:
            self.shortcut = nn.Sequential()


    def forward(self, x):
        output1 = self.relu(self.batchnorm1(self.conv1(x)))
        output2 = self.relu(self.batchnorm2(self.conv2(output1)))
        output3 = self.relu(self.batchnorm3(self.conv3(output2)))

        shortcut = self.shortcut(x)

        return nn.ReLU(inplace=True)(output3 + shortcut)


class ResNet(nn.Module):

    def __init__(self, blockType, numChannels, numConv2Duplicates, numConv3Duplicates, numConv4Duplicates, numClasses):
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(numChannels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.conv2 = self._make_layer(blockType, 16, 16, numConv2Duplicates, 1)
        self.conv3 = self._make_layer(blockType, 16, 32, numConv3Duplicates, 2)
        self.conv4 = self._make_layer(blockType, 32, 64, numConv4Duplicates, 2)               
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, numClasses)

    def _make_layer(self, block, in_size, out_size, num_blocks, stride):              
        layers = []
        layers.append(BasicBlock(in_size, out_size, stride))
        for _ in range(num_blocks-1):
            layers.append(BasicBlock(out_size, out_size))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)  
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(train_loader, model, criterion, optimizer, device, PATH, numEpochs, FILE, useWarmup, learning_rate, STATE_PATH):
    
    f = open(FILE, "w")
    print_freq = 10
    model.train()    
    currEpoch = 0
    total_step = len(train_loader)
    numIterations = 0
    totalIterations = 64000
    while numIterations < totalIterations:    
        for i, (images, labels) in enumerate(train_loader):

            #with torch.autograd.detect_anomaly():
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            loss = criterion(output, labels)

            acc1, acc5 = accuracy(output, labels, topk=(1,5))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_freq == 0:
                updateString = "Epoch:{} TotalEpochs:{} Itr:{} Step:[{}/{}] Loss:{:.4f} \n".format(currEpoch+1, numEpochs, numIterations+1, i+1, total_step, loss.item())
                f.write(updateString)
                print(updateString)    

            numIterations += 1

            if useWarmup and numIterations == 500:                
                for g in optimizer.param_groups:
                    print("Setting learning rate to 0.01")                    
                    learning_rate = 0.01
                    g['lr'] = learning_rate

            if numIterations == 32000:                
                for g in optimizer.param_groups:                    
                    learning_rate = learning_rate/10
                    g['lr'] = learning_rate

            if numIterations == 48000:
                learning_rate = learning_rate/10                
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate   

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

            updateString = 'NumIterations: {} Epoch Accuracy of the model on the test images: {} % \n'.format( numIterations, 100 * correct / total) 
            print(updateString)
            f.write(updateString)

        torch.save({
            'epoch': currEpoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss            
            }, PATH)
        torch.save(model.module.state_dict(), STATE_PATH)
        
        currEpoch += 1

def getParams(model):
    param_counts = [np.prod(p.size()) for p in model.parameters()]
    numParams = sum(param_counts)
    return numParams

numClasses = 10 
learning_rate = 0.1
#learning_rate=0.001 #Should usually be 0.1, but use this for n=18. Meant to be warmup rate but doesn't work
useWarmup = False
loadModel = False
bs = 128
numEpochs = 150

cropSize = 32
numChannels=3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n = 3

FILE = 'resnet_n={}.txt'.format(n)
PATH = 'resnet_n={}.ckpt'.format(n)
STATE_PATH = 'n={}_state_dict.ckpt'.format(n)

model = ResNet(BasicBlock, numChannels, n, n, n, numClasses).to(device)

numParams = getParams(model)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

if loadModel == True:
    model.load_state_dict(torch.load(PATH))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='cifar-10-data/',
                                             train=True, 
                                             transform=transform_train,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='cifar-10-data/',
                                            train=False, 
                                            transform=transform_test)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=bs, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=bs, 
                                          shuffle=False)

train(train_loader, model, criterion, optimizer, device, PATH, numEpochs, FILE, useWarmup, learning_rate, STATE_PATH)