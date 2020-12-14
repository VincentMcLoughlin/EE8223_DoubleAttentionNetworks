import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import math
from torchsummary import summary
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from PIL import Image

class IdentityPadding(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()        
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
            self.projection = IdentityPadding(out_size, in_size, stride)
        else:
            self.projection = None
        
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_size)
        self.relu1 = nn.ReLU(inplace=True) #ReLU performed after addition, should I use inPlace?
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_size) 
        self.relu2 = nn.ReLU(inplace=True)        

    def forward(self, x): #Will need shortcut case

        res = x       
        output = self.conv1(x)
        output = self.batchnorm1(output)
        output = self.relu1(output)        
        output = self.conv2(output)
        output = self.batchnorm2(output)

        if self.projection:
            res = self.projection(x)

        output += res
        out = self.relu2(output)

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
        #self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, numClasses)

    def _make_layer(self, block, in_size, out_size, num_blocks, stride):              
        layers = []
        layers.append(BasicBlock(in_size, out_size, stride))
        for _ in range(num_blocks-1): #Double check this
            layers.append(BasicBlock(out_size, out_size))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        #print(output.size())
        output = self.conv2(output)
        #print(output.size())
        output = self.conv3(output)
        #print(output.size())
        output = self.conv4(output)
        #print(output.size())        
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def image_loader(image_path):
    imsize = 32
    loader = transforms.Compose([
        transforms.Scale(imsize), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    testimage  = image.open(image_path)
    testimage  = loader(testimage).float()
    testimage = Variable(testimage, requires_grad=True)
    return testimage

class WrappedModel(nn.Module):
	def __init__(self, module):
		super(WrappedModel, self).__init__()
		self.module = module # that I actually define.
	def forward(self, x):
		return self.module(x)


model = getattr(models, args.model)(args)
model = WrappedModel(model)
state_dict = torch.load(modelname)['state_dict']
model.load_state_dict(state_dict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n = 7 #90.39% accuracy with n=7
numChannels = 3
numClasses = 10
#took 3 tries with n=9
#Took 3 tries with n=9
#Took like 6 with n=7
#n=5 worked once out of like 10 times 
#n=3 never worked
#FILE = 'resnet_n={}.txt'.format(n)
PATH = 'resnet_n={}_test.ckpt'.format(n)

print(PATH)

device = 'cpu'
model = ResNet(BasicBlock, numChannels, n, n, n, numClasses)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(PATH, map_location=device), strict=False)
model.eval()

trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

IMAGE_PATH="car.jpg"
image = Image.open(IMAGE_PATH)
inputImage = trans(image)

inputImage = inputImage.view(1, 3, 32,32)

output = model(inputImage)
print(output)
prediction = int(torch.max(output.data, 1)[1].numpy())
print(prediction)