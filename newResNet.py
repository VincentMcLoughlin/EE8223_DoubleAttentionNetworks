import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import math
from torchsummary import summary

class InitialConvblock(nn.Module): #Double check this with the paper
    def __init__(self, in_size, out_size, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, stride=stride, kernel_size=7, padding=1) #SHOULD STRIDE BE 2, KERNEL 7?
        self.batchnorm1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def InitialConvblock(self, x):
        output = self.pool(self.pool(self.relu(self.batchnorm1(self.conv1(x)))))        
        return output

    def forward(self, x):       
        return InitialConvblock(x)




class BasicBlock(nn.Module):
    def __init__(self, in_size, out_size, stride=1): #May need resizing parameter
        super().__init__()

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True) #ReLU performed after addition, should I use inPlace?
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=stride, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_size) 


    def forward(self, x): #Will need shortcut case
        output = self.conv1(x)
        output = self.batchnorm1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.batchnorm2(output)
        output = output + x
        output = self.relu(output) #According to original paper, relu performed after addition
        return output


class BottleneckBlock(nn.Module):
    def __init__(self, in_size, out_size, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=1, bias=False) #Should bias be true?
        self.batchnorm1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, bias=False, padding=1, stride=stride) #Do we need padding?
        self.batchnorm2 = nn.BatchNorm2d(out_size)
        #Relu again

        self.conv3 = nn.Conv2d(out_size, out_size*4, kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_size)
        #Relu Again

        if stride != 1 or in_size != out_size*4: #Is this not always true, looks wrong. Nvm, with makelayer this is actually smart
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size*4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size*4) #May need to set batchnorm weights to 0
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

    def __init__(self, blockType, numChannels, numConv2Duplicates, numConv3Duplicates, numConv4Duplicates, numConv5Duplicates, numClasses):
        super().__init__()

        self.conv1 = InitialConvblock(numChannels, 32, stride=2)
        self.conv2 = self.makeLayer(blockType, numConv2Duplicates, in_size=32, out_size=64)
        self.conv3 = self.makeLayer(blockType, numConv3Duplicates, in_size=32, out_size=128)
        self.conv4 = self.makeLayer(blockType, numConv3Duplicates, in_size=32, out_size=256)
        self.conv5 = self.makeLayer(blockType, numConv4Duplicates, in_size=32, out_size=512)
        self.globalAvg = nn.AvgPool2d(kernel_size=4, stride=1) #stride should be size of feature map (8?)
        self.fc = nn.Linear(512, numClasses)


    def makeLayer(self, block, numDuplicates, in_size, out_size, initialStride=1, ):
        resnetLayer = []

        firstBlock = block(in_size, out_size, initialStride)

        resnetLayer.append(firstBlock)

        for i in range(1,numDuplicates):
            resnetLayer.append(block(out_size, out_size))

        finishedResnetLayer = nn.Sequential(*resnetLayer)

        return finishedResnetLayer

    def forward(self, x):
        return x


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
def train(train_loader, model, criterion, optimizer, device, args):
    
    print_freq = 10
    model.train()
    for currEpoch < numEpochs:    
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            acc1, acc5 = accuracy(output, labels, topk=(1,5))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_freq == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(currEpoch+1, num_epochs, i+1, total_step, loss.item()))





#model = ResNet(BasicBlock, 3, 2, 2, 2, 2, 10)
numClasses = 200
model = ResNet(BottleneckBlock, 3, 3, 4, 6, 3, numClasses)
x = torch.ones(1,3,224,224)
z = model(x)

transform_train = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4), #Need to change these when doing imagenet
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.3)
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.3)
    ])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR100(root='../cifar-100-data/',
                                             train=True, 
                                             transform=transform_train,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR100(root='../cifar-100-data/',
                                            train=False, 
                                            transform=transform_test)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=bs, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=bs, 
                                          shuffle=False)

train(train_loader, model, criterion, optimizer, args)
