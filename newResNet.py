import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import math
from torchsummary import summary
from torch.autograd import Variable
import torch.nn.functional as F


# Not written by me, obtained from https://github.com/nguyenvo09/Double-Attention-Network/blob/master/double_attention_layer.py
class DoubleAttentionLayer(nn.Module):
    """
    Implementation of Double Attention Network. NIPS 2018
    """
    def __init__(self, in_channels: int, c_m: int, c_n: int, reconstruct = False):
        """
        Parameters
        ----------
        in_channels
        c_m
        c_n
        reconstruct: `bool` whether to re-construct output to have shape (B, in_channels, L, R)
        """
        super(DoubleAttentionLayer, self).__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size = 1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: `torch.Tensor` of shape (B, C, H, W)
        Returns
        -------
        """
        batch_size, c, h, w = x.size()
        assert c == self.in_channels, 'input channel not equal!'
        A = self.convA(x)  # (B, c_m, h, w) because kernel size is 1
        B = self.convB(x)  # (B, c_n, h, w)
        V = self.convV(x)  # (B, c_n, h, w)
        tmpA = A.view(batch_size, self.c_m, h * w)
        attention_maps = B.view(batch_size, self.c_n, h * w)
        attention_vectors = V.view(batch_size, self.c_n, h * w)
        attention_maps = F.softmax(attention_maps, dim = -1)  # softmax on the last dimension to create attention maps
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim = 1)  # (B, c_n, h * w) attention on c_n dimension
        tmpZ = global_descriptors.matmul(attention_vectors)  # B, self.c_m, h * w
        tmpZ = tmpZ.view(batch_size, self.c_m, h, w)
        if self.reconstruct: tmpZ = self.conv_reconstruct(tmpZ)
        return tmpZ


class InitialConvblock(nn.Module): #Double check this with the paper
    def __init__(self, in_size, out_size, stride=2, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, stride=stride, kernel_size=kernel_size, padding=2) #SHOULD STRIDE BE 2, KERNEL 7?
        self.batchnorm1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)    

    def forward(self, x):       
        return self.pool(self.relu(self.batchnorm1(self.conv1(x))))




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

    def __init__(self, blockType, numChannels, numConv2Duplicates, numConv3Duplicates, numConv4Duplicates, numConv5Duplicates, numClasses, c_n, numAttentionDuplicates):
        super().__init__()

        self.conv1 = InitialConvblock(in_size=numChannels, out_size=32, stride=2)
        self.conv2 = self.makeLayer(blockType, numConv2Duplicates,1, in_size=32, out_size=64)
        self.conv3 = self.makeLayer(blockType, numConv3Duplicates,2, in_size=64*4, out_size=128)
        self.doubleABlock3 = self.makeAttentionLayer(numAttentionDuplicates, c_n, in_size=128*4, out_size=128*4)
        self.conv4 = self.makeLayer(blockType, numConv4Duplicates,2, in_size=128*4, out_size=256)
        self.doubleABlock4 = self.makeAttentionLayer(numAttentionDuplicates, c_n, in_size=256*4, out_size=256*4)
        self.conv5 = self.makeLayer(blockType, numConv5Duplicates,2, in_size=256*4, out_size=512)        
        self.globalAvg = nn.AdaptiveAvgPool2d((1,1)) #Is this better?
        self.fc = nn.Linear(2048, numClasses)


    #def makeLayer(self, block, numDuplicates, in_size, out_size, initialStride=1):
    def makeLayer(self, block, numDuplicates, initialStride, in_size, out_size): #Problem with this layer

        layers = []
        stride=1
        in_channels = in_size
        firstBlock = block(in_channels,out_size,initialStride)
        layers.append(firstBlock)
        in_channels = out_size*4
        for i in range(1, numDuplicates):
            layers.append(block(in_channels, out_size, stride))
            in_channels = out_size * 4
        
        return nn.Sequential(*layers)

    def makeAttentionLayer(self, numDuplicates, c_n, in_size, out_size, stride=1):
        attentionLayer = []

        for i in range(0,numDuplicates):
            attentionLayer.append(DoubleAttentionLayer(in_size, out_size, c_n, reconstruct=True))

        finishedLayer = nn.Sequential(*attentionLayer)

        return finishedLayer

    def forward(self, x):
        
        output = self.conv1(x)        
        output = self.conv2(output)
        output = self.conv3(output)        
        #output = self.doubleABlock3(output)
        output = self.conv4(output)
        #output = self.doubleABlock4(output)
        output = self.conv5(output)        
        output = self.globalAvg(output)
        
        output = output.view(output.size(0),-1)
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

def train(train_loader, model, criterion, optimizer, device, args, PATH):
    
    print_freq = 10
    model.train()
    numEpochs = 120
    currEpoch = 0
    while currEpoch < numEpochs:    
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

        torch.save({
            'epoch': currEpoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)
        
        currEpoch += 1



#model = ResNet(BasicBlock, 3, 2, 2, 2, 2, 10)
numClasses = 200
learning_rate = 0.0001
bs = 128
numEpochs = 100
c_n = 4
numAttentionDuplicates = 5
cropSize = 224
numChannels=3
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet(BottleneckBlock, numChannels, 3, 4, 6, 3, numClasses, c_n, numAttentionDuplicates) #.to(device)

# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = nn.DataParallel(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x = torch.ones(1,3,224,224)
z = model(x)
print(z.size())
print(model)
summary(model.cuda(), (3, 224, 224))

exit()
transform_train = transforms.Compose([
    #transforms.ToPILImage(),
    #transforms.RandomCrop(cropSize, padding=4), #Need to change these when doing imagenet
    transforms.RandomCrop(cropSize),
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

#train(numEpochs, train_loader, model, criterion, optimizer, args, PATH)

