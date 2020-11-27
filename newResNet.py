import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import math
from torchsummary import summary

class InitialConvblock(nn.Module): #Double check this with the paper
    def __init__(self, in_size, out_size, stride=2):
        super(InitialConvblock, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, stride=stride, kernel_size=7, padding=1) #SHOULD STRIDE BE 2, KERNEL 7?
        self.batchnorm1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def InitialConvblock(self, x):
        output = self.pool(self.pool(self.relu(self.batchnorm1(self.conv1(x)))))        
        return output

    def forward(self, x):       
        return InitialConvblock(x)




class BasicBlock(nn.Module):
    def __init__(self, in_size, out_size, stride=1): #May need resizing parameter
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU() #ReLU performed after addition, should I use inPlace?
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


class ResNet(nn.Module):

    def __init__(self, blockType, numChannels, numConv2Duplicates, numConv3Duplicates, numConv4Duplicates, numConv5Duplicates, numClasses):
        

        self.conv1 = InitialConvblock(numChannels, 32, stride=2)
        self.conv2 = self.makeLayer(blockType, numConv2Duplicates, in_size=32, out_size=64)
        self.conv3 = self.makeLayer(blockType, numConv3Duplicates, in_size=32, out_size=128)
        self.conv4 = self.makeLayer(blockType, numConv3Duplicates, in_size=32, out_size=256)
        self.conv5 = self.makeLayer(blockType, numConv4Duplicates, in_size=32, out_size=512)
        self.globalAvg = nn.AvgPool2d(kernel_size=4, stride=1) #stride should be size of feature map (8?)
        self.fc = nn.Linear(512, numClasses)


    def makeLayer(self, block, numDuplicates, initialStride, in_size, out_size):
        resnetLayer = []

        firstBlock = block(in_size, out_size, stride)

        resnetLayer.append(firstBlock)

        for i in range(1,numDuplicates):
            resnetLayer.append(block(out_size, out_size))

        finishedResnetLayer = nn.Sequential(*resnetLayer)

        return finishedResnetLayer

    def forward(self, x):
        return x

model = ResNet(BasicBlock, 3, 2, 2, 2, 2, 10)
print(mode)