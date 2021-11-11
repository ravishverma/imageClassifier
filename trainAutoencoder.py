import torch
from modelLib import *
from environment import *
from dataLoader import *
import os
from matplotlib import pyplot as plt
from pytorch_model_summary import summary

# Initialize data loader
dataset_train = myDataset(path=DATADIR,source='train',val_split=0.2)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCHSIZE, num_workers=4, pin_memory=True)

dataset_val = myDataset(path=DATADIR,source='validation',val_split=0.2)
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=BATCHSIZE, num_workers=4, pin_memory=True)

# Initialize ML Model
model = autoencoder().to(DEVICE)
print(summary(model, torch.zeros(1,1,64,64)))

reconLoss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

lossTrain = []
lossVal = []
if EPOCH0>0:
    model, optimizer, _, lossTrain, lossVal = loadChkPt(f'chkpt_{EPOCH0-1}.pt', model, optimizer)

# Train
for epoch in range(NEPOCHS):
    print(f'Epoch: {EPOCH0+epoch}')
    lossTrain_s = 0
    lossVal_s = 0
    
    model.train()
    for i, imgs in enumerate(loader_train):
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()
        _, reconImgs = model(imgs)
        loss = reconLoss(reconImgs, imgs)
        lossTrain_s += loss.item()
        loss.backward()
        optimizer.step()  

    # Validation
    model.eval()
    with torch.no_grad():
        for i, imgs in enumerate(loader_val):
            imgs = imgs.to(DEVICE)
            _, reconImgs = model(imgs)
            loss = reconLoss(reconImgs, imgs)
            lossVal_s+=loss.item()

    lossTrain.append(lossTrain_s/len(loader_train))
    lossVal.append(lossVal_s/len(loader_val))

    if (EPOCH0+epoch+1)%SAVEINTERVAL==0:
        chkpt = {'epoch': EPOCH0+epoch,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'lossTrain': lossTrain,
                 'lossVal': lossVal}
        saveChkPt(chkpt, f'chkpt_{EPOCH0+epoch}.pt')

        print(f'Train Loss: {lossTrain[-1]}, Val Loss: {lossVal[-1]}')

        plt.figure()
        plt.plot(range(EPOCH0+epoch+1), lossTrain, label='Train')
        plt.plot(range(EPOCH0+epoch+1), lossVal, label='Validation')
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.savefig(os.path.join(RESULTSDIR,"trainAutoencoder.png"))
        plt.close()
