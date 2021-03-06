import torch.nn as nn
import torch.optim as optim
from environment import *

# Input: Grayscale images: 64 x 64 x 1

# Reduce image size to 1/4 x 1/4
class encodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.last = last

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)
        return x

# Expand image size to 4 x 4
class decodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last=False):
        super().__init__()
        self.convTrans1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.convTrans2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.last = last

    def forward(self,x):
        x = self.convTrans1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.convTrans2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x) if not self.last else x
        return x

# Autoencoder
class autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = encodeBlock(1,20) # 64x64 > 16x16
        self.down2 = encodeBlock(20,40) # 16x16 > 4x4
        self.down3 = encodeBlock(40,20) # 4x4 > 1x1

        self.fcn1 = nn.Linear(20,LSSIZE)
        self.fcn2 = nn.Linear(LSSIZE,20)
        self.relu = nn.ReLU()

        self.up1 = decodeBlock(20,40) 
        self.up2 = decodeBlock(40,20) 
        self.up3 = decodeBlock(20,1, last=True) 

    def forward(self,x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        x = x.view(-1,20)
        ls = self.fcn1(x)
        x = self.relu(ls)
        x = self.relu(self.fcn2(x))
        x = x.view(-1,20,1,1)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return ls, x
