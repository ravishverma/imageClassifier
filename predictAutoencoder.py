import torch
from dataLoader import *
from environment import *
from modelLib import *

class predictAE:
    def __init__(self, chkptFile):
        self.model = autoencoder()
        self.model, _, _, _, _ = loadChkPt(chkptFile, self.model) 
        self.dataset = myDataset()

    def reduce(self, filepath):
        img = self.dataset.loadFromFile(filepath)

        self.model.eval()

        with torch.no_grad():
            ls, _ = self.model(img)
            return ls

def assessAE(chkptFile):
    from matplotlib import pyplot as plt

    predict = predictAE(chkptFile)

    dataset = myDataset(path=DATADIR)

    lsRepX = []
    lsRepY = []
    for f in dataset.files:
        ls = predict.reduce(os.path.join(DATADIR,f))
        lsRepX.append(ls[0].float())
        lsRepY.append(ls[1].float())

    fig = plt.figure()
    plt.scatter(lsRepX, lsRepY)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(os.path.join(RESULTSDIR,"assessLatentSpace.png"))    

if __name__=="__main__":
    import sys
    chkptFile = sys.argv[1]
    assessAE(chkptFile)
