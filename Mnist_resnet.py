import pickle
import gzip
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset
import torch.utils.data as data

from datetime import datetime
import math
import torchvision
import time
from torchsummary import summary
import torchvision.models as models

# initialOutputSize = 56
# #block1Size = 32 TOD Change back
# block1Size = 64
# convBlock2Size = 64
# convBlock3Size = 128
# convBlock4Size = 256
# convBlock5Size = 512

#MNIST

#block1Size = 32 TOD Change back
initialOutputSize = 28
block1Size = 16
convBlock2Size = 16
convBlock3Size = 16
convBlock4Size = 16
convBlock5Size = 16

def get_data_batches(x_train, y_train, x_val, y_val, bs):
  train_ds = TensorDataset(x_train, y_train)
  val_ds = TensorDataset(x_val, y_val) 
  return (
      DataLoader(train_ds, batch_size=bs, shuffle=True),
      DataLoader(val_ds, batch_size=bs * 2),
  )

def get_files(path, filename):
  with gzip.open((PATH+FILENAME), "rb") as file:
    ((x_train, y_train), (x_val, y_val), _) = pickle.load(file, encoding='latin-1')
  return x_train, y_train, x_val, y_val

#Need to map numpy arrays in data to tensor, this function applies torch.tensor to the values passed in
def tensor_map(x_train, y_train, x_val, y_val): return map(torch.cuda.FloatTensor, (x_train, y_train, x_val, y_val))

def preprocess(x): #Little preprocessing function to reshape according to our input, parameters are x.size(batch size, channels (1 for greyscale), height, width)
  return x.view(-1, 1, 28, 28) #-1 means take whatever dimension is passed in and use that, alows variable batch size

def outSizeCalculator(inSize, padding, kernelSize, stride):
  return floor((inSize+2*padding - kernelSize)/stride + 1)

class InitialConvblock(nn.Module):
  def __init__(self, in_size, out_size): #In size of 224, outsize of 56
    super().__init__()
    self.kernelSize = 5
    self.stride = 2
    self.pad = 2
    self.conv1 = nn.Conv2d(in_size, out_size, self.kernelSize, self.stride, self.pad)
    self.batchnorm1 = nn.BatchNorm2d(out_size)
    self.relu = nn.ReLU(inplace=True)    
    self.maxpool = nn.AdaptiveMaxPool2d(initialOutputSize)

  def InitialConvblock(self, x):    
    x = self.maxpool(self.relu(self.batchnorm1(self.conv1(x))))
    return x

  def forward(self, x): return self.InitialConvblock(x)

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
      #a = 0
    else:
      shortcut = self.batchnormShortcut(self.convShortcut(x))
      #a = 0
    blockOutput = self.relu(self.batchnorm3(self.conv3(self.relu(self.batchnorm1(self.conv2(self.relu(self.batchnorm1(self.conv1(x)))))))))
    #blockOutput = self.relu(self.batchnorm2(self.conv1(x)))

    return shortcut + blockOutput
    #return blockOutput

  def forward(self, x, resizeShortcut=False): return self.resblock(x, resizeShortcut)


class ResNet(nn.Module):

  def __init__(self, n_classes=1000):
    super().__init__()  

    self.nClasses = n_classes
    #self.convBlock1 = InitialConvblock(numChannels, block1Size)
    self.convBlock2 = InitialConvblock2(numChannels, block1Size)
  
    self.block2a = ResBlock(block1Size, convBlock2Size, convBlock2Size*4)
    self.block2b = ResBlock(convBlock2Size*4, convBlock2Size,convBlock2Size*4)
    
    self.block3a = ResBlock(convBlock2Size*4, convBlock3Size, convBlock3Size*4, True)
    self.block3b = ResBlock(convBlock3Size*4, convBlock3Size, convBlock3Size*4)

    self.block4a = ResBlock(convBlock3Size*4, convBlock4Size, convBlock4Size*4, True)
    self.block4b = ResBlock(convBlock4Size*4, convBlock4Size, convBlock4Size*4)

    self.block5a = ResBlock(convBlock4Size*4, convBlock5Size, convBlock5Size*4, True)
    self.block5b = ResBlock(convBlock5Size*4, convBlock5Size, convBlock5Size*4)

    self.globalAvgPool = nn.AdaptiveAvgPool2d(1) #Kernel Size used to be 7
    self.linearLayer = nn.Linear(block1Size*4, self.nClasses) #num classes
    self.relu = nn.ReLU(inplace=True)


  def forward(self, x):
    x = preprocess(x)
    #print(x.size())
    x = self.convBlock2(x)
    #print(x.size())
    #x = self.convBlock2(x)
    
    x = self.block2b(self.block2b(self.block2a(x)))
    #print(x.size())
    x = self.block3b(self.block3b(self.block3b(self.block3a(x, True))))
    #print(x.size())
    x = self.block4b(self.block4b(self.block4b(self.block4b(self.block4b(self.block4a(x, True))))))
    #print(x.size())
    x = self.block5b(self.block5b(self.block5a(x, True)))
    #print(x.size())
    x = self.globalAvgPool(x)
    x = x.view(-1, block1Size*4)
    x = self.relu(self.linearLayer(x))
    #print(x.size())
    return x

def loss_batch(model, loss_func, xb, yb, opt=None, scheduler=None):  

  loss = loss_func(model(xb), yb.long())
  acc = accuracy(model(xb), yb)
  if opt is not None:
      loss.backward()
      if scheduler is not None:
        scheduler.step()
      opt.step()
      opt.zero_grad()
  
  return acc, loss.item(), len(xb)

def accuracy(out, yb):  
  preds = torch.argmax(out, dim=1)
  return (preds == yb).float().mean() #Need to convert to float for mean to work

def train_model(epochs, model, loss_func, opt, train_dl, test_dl, device, savePath, scheduler=None):
  for epoch in range(epochs):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Beginning Epoch at ", current_time)
    
    model.train()
    i = 0
    running_loss = 0
    displayInterval = 50

    for inputs, labels, in train_dl: #xb and yb could be wrong size? No doesn't look like it
      #print(inputs.size())
      #print(yb.size())
      inputs = preprocess(inputs)
      #print(inputs.size())
      inputs = inputs.to(device)
      labels = labels.to(device)
      loss, acc, num = loss_batch(model, loss_func, inputs, labels, opt, scheduler)
      running_loss += loss

      if i % displayInterval == displayInterval - 1:
        print("Batch Number: ", i)
        currLoss = running_loss/displayInterval
        print("Running Loss: ", currLoss)
        running_loss = 0
      i += 1

    with torch.no_grad():
      accs, losses, nums = zip(
          *[loss_batch(model, loss_func, xb, yb) for xb, yb in test_dl]
      )

      val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
      val_acc = np.sum(np.multiply(accs, nums)) / np.sum(nums)

      print("Epoch:", epoch+1)
      print("Loss: ", val_loss)
      print("Accuracy: ", val_acc)
      print()

    # print("Evaluating Model")
    # model.eval()
    # #No gradient computation for evaluation mode
    # with torch.no_grad():
    #   accs = []
    #   losses = []
    #   nums = []
    #   for xb, yb in test_dl:
    #     acc, loss, num = loss_batch(model, loss_func, xb.to(device), yb.to(device))
    #     accs.append(acc)
    #     losses.append(loss)
    #     nums.append(num) 
      
       
    #   val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    #   val_acc = np.sum(np.multiply(accs, nums)) / np.sum(nums)

    #   print("Epoch:", epoch+1)
    #   print("Loss: ", val_loss)
    #   print("Accuracy: ", val_acc)
    #   torch.save(model.state_dict(), savePath)


def get_train_dataset(dir):
  imageSize = 5
  trainTransforms = transforms.Compose([
    transforms.RandomResizedCrop(imageSize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.455, 0.405], std=[0.229, 0.224, 0.225]) #Check if std deviation should be set
  ])

  trainDataset = datasets.ImageFolder(dir, trainTransforms)

  return trainDataset

def get_val_dataset(dir):
    
  valTransforms = transforms.Compose([    
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.455, 0.405], std=[0.229, 0.224, 0.225]) #Check if std deviation should be set
  ])
  valDataset = datasets.ImageFolder(dir, valTransforms)
  
  return valDataset

def getDataloaders(dataDir, valDir, batchSize, workers=0, pin_memory=False): #Can also add pin_memory 
  trainDataset = get_train_dataset(dataDir)
  valDataset = get_val_dataset(valDir)

  trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batchSize, shuffle=True, num_workers=workers, pin_memory=pin_memory, sampler=None)
  valLoader = torch.utils.data.DataLoader(valDataset, batch_size = batchSize, shuffle=False, num_workers=workers, pin_memory=pin_memory)
  return trainLoader, valLoader

#Hyperparameters
bs = 128 #Max batch size is 128
lr = math.sqrt(0.1)
n_epochs = 2
loss_func = F.cross_entropy
#numChannels = 3
#numClasses = 200
numChannels = 1
numClasses = 10
outputFile = '/home/mcvi0001/A^2_Net_Code/Imagenet_resnet.pth'

#trainDir = "/home/mcvi0001/A^2_Net_Code/tiny-imagenet-200/train"
#testDir = "/home/mcvi0001/A^2_Net_Code/tiny-imagenet-200/test"
#valDir = "/home/mcvi0001/A^2_Net_Code/tiny-imagenet-200/val"

# trainDir = "/content/drive/My Drive/EE Deep Learning/tiny-imagenet-200/train"
# testDir = "/content/drive/My Drive/EE Deep Learning/tiny-imagenet-200/test"
#TINYIMAGENET DATASET
# trainDir = "/content/drive/My Drive/EE Deep Learning/tiny-imagenet-200/train"
# testDir = "/content/drive/My Drive/EE Deep Learning/tiny-imagenet-200/val"
# trainDL, testDL = getDataloaders(trainDir, testDir, bs)

#MNIST DATASET 
PATH = "/home/mcvi0001/A^2_Net_Code/"
FILENAME = 'mnist.pkl.gz'

#get data set
x_train, y_train, x_val, y_val = get_files(PATH, FILENAME)

# map tensor function to all inputs (X) and targets (Y) to create tensor data sets
x_train, y_train, x_val, y_val = tensor_map(x_train, y_train, x_val, y_val)

# get math.ceil(x_train.shape[0]/batch_size) train and val mini batches of size bs
trainDL, testDL = get_data_batches(x_train, y_train, x_val, y_val, bs)
#########

print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Model/optimizer
model = ResNet(numClasses)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

model.to(device)
#print(model)
#summary(model.cuda(), (3, 224, 224))
#summary(models.resnet50(False).cuda(), (3, 224, 224))
#train 
train_model(n_epochs, model, loss_func, optimizer, trainDL, testDL, device, outputFile)
input = torch.ones(1, 1, 28, 28)
#model(input)
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Finished Training at", current_time)
torch.save(model.state_dict(), outputFile)
print("Finished Saving")
#PATH = './data/mnist/'