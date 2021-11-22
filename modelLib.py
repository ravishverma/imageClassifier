import torch.nn as nn
import torch.optim as optim

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
        x = self.batchNorm(x)
        x = self.relu(x) if not self.last else x
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
        x = self.conv1(x)
        x = self.relu(x)
        x = self.convTrans2(x)
        x = self.conv2(x)
        x = self.relu(x) if not self.last else x
        return x

# Autoencoder
class autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = encodeBlock(1,20) # 64x64 > 16x16
        self.down2 = encodeBlock(20,40) # 16x16 > 4x4
        self.down3 = encodeBlock(40,60) # 4x4 > 1x1
        self.lin1 = nn.Linear(60,10)
        self.lin2 = nn.Linear(10,3)
        self.lin3 = nn.Linear(3,10)
        self.lin4 = nn.Linear(10,60)
        self.up1 = decodeBlock(60,40) 
        self.up2 = decodeBlock(40,20) 
        self.up3 = decodeBlock(20,1) 
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        x = x.reshape((-1,60))
        x = self.relu(self.lin1(x))
        ls = self.lin2(x)
        x = self.relu(ls)
        x = self.relu(self.lin3(x))
        x = self.relu(self.lin4(x))
        x = x.reshape((-1,60,1,1))

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return ls, x     
