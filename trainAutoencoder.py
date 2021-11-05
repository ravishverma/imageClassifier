import torch
from modelLib import *
from environment import *
from dataLoader import *
import os
from matplotlib import pyplot as plt

# Initialize data loader
batchSize = 1

dataset_train = myDataset(os.path.join(dataDir,'trainData'))
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batchSize, shuffle=True, num_workers=2, pin_memory=True)

dataset_val = myDataset(os.path.join(dataDir,'valData'))
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batchSize, shuffle=True, num_workers=2, pin_memory=True)

# Initialize training parameters
epoch0 = 0
nEpochs = 50
saveInterval = 10

# Initialize ML Model
model = autoencoder()
reconLoss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

lossTrain = []
lossVal = []
if epoch0>0:
    model, optimizer, _, loss_train, loss_val = loadChkPt(f'chkpt_{epoch0-1}.pt', model, optimizer)

# Train
for epoch in range(nEpochs):
    print(f'Epoch: {epoch0+epoch}')
    loss_train_s = 0
    loss_val_s = 0
    
    model.train()
    for i, imgs in enumerate(loader_train):
        optimizer.zero_grad()
        reconImgs = model(imgs)
        loss = reconLoss(reconImgs, imgs)
        loss_train_s += loss.item()
        loss.backward()
        optimizer.step()  

    # Validation
    model.eval()
    with torch.no_grad():
        for i, imgs in enumerate(loader_val):
            reconImgs = model(imgs)
            loss = reconLoss(reconImgs, imgs)
            loss_val_s+=loss.item()

    loss_train.append(loss_train_s/len(loader_train))
    loss_val.append(loss_val_s/len(loader_val))

    if (epoch0+epoch+1)%saveInterval==0:
        chkpt = {'epoch': epoch0+epoch,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'loss_train': loss_train,
                 'loss_val': loss_val}
        saveChkPt(chkpt, f'chkpt_{epoch0+epoch}.pt')

        print(f'Train Loss: {loss_train[-1]}, Val Loss: {loss_val[-1]}')

        plt.figure()
        plt.plot(range(epoch0+epoch+1), loss_train, label='Train')
        plt.plot(range(epoch0+epoch+1), loss_val, label='Validation')
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.savefig(os.path.join(resultsDir,"train.png"))
        plt.close()