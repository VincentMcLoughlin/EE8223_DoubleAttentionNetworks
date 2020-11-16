import pickle
import gzip
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from google.colab import drive
from datetime import datetime
drive.mount('/content/drive/')


#Resnet following this guide https://github.com/LukasMut/ResNet-in-PyTorch/blob/master/ResNet%20in%20PyTorch.ipynb
#Mnist dataset located in data folder in same folder as this notebook
#Made for google colab

def get_files(path, filename):
  with gzip.open((PATH+FILENAME), "rb") as file:
    ((x_train, y_train), (x_val, y_val), _) = pickle.load(file, encoding='latin-1')
  return x_train, y_train, x_val, y_val

#Need to map numpy arrays in data to tensor, this function applies torch.tensor to the values passed in
def tensor_map(x_train, y_train, x_val, y_val): return map(torch.cuda.FloatTensor, (x_train, y_train, x_val, y_val))

def preprocess(x): #Little preprocessing function to reshape according to our input, parameters are x.size(batch size, channels (1 for greyscale), height, width)
  return x.view(-1, 1, 28, 28) #-1 means take whatever dimension is passed in and use that, alows variable batch size

#In order not to call nn.Conv2d multiple times, we write this wrapper function
def conv(in_size, out_size, pad=1):
  return nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=pad)

class ResBlock(nn.Module): #Define our resblock 
  def __init__(self, in_size:int, hidden_size:int, out_size:int, pad:int):
      super().__init__()
      self.conv1 = conv(in_size, hidden_size, pad)
      self.conv2 = conv(hidden_size, out_size, pad)
      self.batchnorm1 = nn.BatchNorm2d(hidden_size)
      self.batchnorm2 = nn.BatchNorm2d(out_size)

  def convblock(self, x):
    x = F.relu(self.batchnorm1(self.conv1(x)))
    x = F.relu(self.batchnorm2(self.conv2(x)))
    return x

  def forward(self, x): return x + self.convblock(x) #Skip the connection
  
class ResNet(nn.Module):

  def __init__(self, n_classes=10):
    super().__init__()
    self.res1 = ResBlock(1, 8, 16, 15)
    self.res2 = ResBlock(16,32, 16, 15)
    self.conv = conv(16, n_classes)
    self.batchnorm = nn.BatchNorm2d(n_classes)
    self.maxpool = nn.AdaptiveMaxPool2d(1) #Adaptive maxpooling allows us to pass inputs of any size

  def forward(self, x):
    x = preprocess(x)
    x = self.res1(x)
    x = self.res2(x)
    x = self.maxpool(self.batchnorm(self.conv(x)))
    return x.view(x.size(0), -1)

#We now write a function to calculate the loss of the mini batch 
#Takes in our resnet, a specified loss function, mini batches for input and target data, 
#and an optional optimizer (needed for training but not for evaluation)
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
  # in Pytorch one cannot take the mean of ints, so they have to be converted to floats
  preds = torch.argmax(out, dim=1)
  return (preds == yb).float().mean()

#get the model
def get_model():
  model = ResNet()
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  return model, optimizer

#get data
#Dataloader is amazing pytorch tool that automatically divides our train and val
#datasets into a set of mini batches based on the given batch size, bs. It can also 
#shuffle the data which is great for training
def get_data_batches(x_train, y_train, x_val, y_val, bs):
  train_ds = TensorDataset(x_train, y_train)
  val_ds = TensorDataset(x_val, y_val) 
  return (
      DataLoader(train_ds, batch_size=bs, shuffle=True),
      DataLoader(val_ds, batch_size=bs * 2),
  )

#Let's fit our model! 
def fit(epochs, model, loss_func, opt, train_dl, valid_dl, scheduler=None):
  for epoch in range(epochs):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Beginning Epoch at ", current_time)
    model.train()
    #iterate over data loader object (generator)
    for xb, yb, in train_dl:
      loss_batch(model, loss_func, xb.to("cuda"), yb.to("cuda"), opt, scheduler)

    model.eval()
    #No gradient computation for evaluation mode
    with torch.no_grad():
      accs, losses, nums = zip(
          *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
      )

      val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
      val_acc = np.sum(np.multiply(accs, nums)) / np.sum(nums)

      print("Epoch:", epoch+1)
      print("Loss: ", val_loss)
      print("Accuracy: ", val_acc)
      print()

#PATH = './data/mnist/'
PATH = "/content/drive/My Drive/EE Deep Learning/mnist/"
FILENAME = 'mnist.pkl.gz'

#Hyperparameters
bs = 64
lr = 0.01
n_epochs = 4
loss_func = F.cross_entropy

#Device
print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#get data set
x_train, y_train, x_val, y_val = get_files(PATH, FILENAME)

# map tensor function to all inputs (X) and targets (Y) to create tensor data sets
x_train, y_train, x_val, y_val = tensor_map(x_train, y_train, x_val, y_val)

# get math.ceil(x_train.shape[0]/batch_size) train and val mini batches of size bs
train_dl, val_dl = get_data_batches(x_train, y_train, x_val, y_val, bs)

#get model and optimizer
model, opt = get_model()
model.to(device)

#train 
fit(n_epochs, model, loss_func, opt, train_dl, val_dl)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")

print("Finished Training at", current_time)
PATH = '/content/drive/My Drive/EE Deep Learning/Mnist_resnet.pth'
torch.save(model.state_dict(), PATH)
print("Finished Saving")