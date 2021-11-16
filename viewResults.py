import pandas as pd
from environment import *
import os
from matplotlib import pyplot as plt
import sys
import torch
import torchvision
from dataLoader import *

dataset = myDataset(path=DATADIR)

df = pd.read_csv(os.path.join(RESULTSDIR,"labeledData.csv"), index_col=0)

uniqueLabels = sorted(df["labels"].unique())

fig, axes = plt.subplots(len(uniqueLabels),1)

for i, uL in enumerate(uniqueLabels):
    sample = df[df["labels"]==uL].sample(n=4)["fileName"].tolist()
    grid = []
    for s in sample:
        img = dataset.loadFromFile(os.path.join(DATADIR,s))
        grid.append(img.reshape(1,64,64))
    gridImg = torchvision.utils.make_grid(grid,nrow=4)
    gridImg = gridImg.permute(1,2,0)
    axes[i].imshow(gridImg)
    axes[i].set_ylabel("Label "+str(uL))

plt.savefig(os.path.join(RESULTSDIR,"results.png"))
plt.close()
