import torch
from torchvision import transforms, utils, datasets
import os
from PIL import Image
from environment import *
import numpy as np

# A custom dataset to generate (Grayscale) image pairs
class myDataset(torch.utils.data.Dataset):
    def __init__(self, path=None, source=None, val_split=0):
        self.path = path

        allFiles = [0] if path==None else os.listdir(path)
        np.random.seed(RANDOMSEED)
        np.random.shuffle(allFiles)

        if DATALIMIT:
            allFiles = allFiles[:DATALIMIT]

        split = int(val_split*(len(allFiles)-1))

        # Filter irrelevant
        if source==None:
             self.files = allFiles
        elif source=="train":
             self.files = allFiles[split:]
        elif source=="validation":
             self.files = allFiles[:split]

        self.transformIn = transforms.Compose([
                     transforms.Grayscale(num_output_channels=1),
                     transforms.Resize(256),
                     transforms.CenterCrop(256),
                     transforms.ToTensor()])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path,self.files[index]))
        inImg = self.transformIn(img)
        return inImg

    def loadFromFile(self, filepath):
        img = Image.open(filepath)
        inImg = self.transformIn(img)
        inImg = np.reshape(inImg, (1,1,256,256))
        return inImg

# save checkpoint
def saveChkPt(state, filename):
    torch.save(state,os.path.join(RESULTSDIR,filename))
    return

# load checkpoint
def loadChkPt(filename, model, optimizer=None):
    chkpt = torch.load(os.path.join(RESULTSDIR,filename))
    model.load_state_dict(chkpt['model'])
    if optimizer!=None: optimizer.load_state_dict(chkpt['optimizer'])
    loss_train = chkpt['lossTrain']
    loss_val = chkpt['lossVal']
    return model, optimizer, chkpt['epoch'], loss_train, loss_val
