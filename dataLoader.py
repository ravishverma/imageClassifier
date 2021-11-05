from torchvision import transforms, util, datasets
import torch
import os
from PIL import Image
from environment import *

# A custom dataset to generate (Grayscale) image pairs
class myDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)
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
        return inImg

# save checkpoint
def saveChkPt(state, filename):
    torch.save(state,os.path.join(resultsDir,filename))
    return

# load checkpoint
def loadChkPt(filename, model, optimizer=None):
    chkpt = torch.load(os.path.join(resultsDir,filename))
    model.load_state_dict(chkpt['model'])
    if optimizer!=None: optimizer.load_state_dict(chkpt['optimizer'])
    loss_train = chkpt['loss_train']
    loss_val = chkpt['loss_val']
    return model, optimizer, chkpt['epoch'], loss_train, loss_val