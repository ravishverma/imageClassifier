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
    import pandas as pd
    import seaborn as sns

    predict = predictAE(chkptFile)

    dataset = myDataset(path=DATADIR)

    lsRep1 = []
    lsRep2 = []
    lsRep3 = []
    for f in dataset.files:
        ls = predict.reduce(os.path.join(DATADIR,f))
        lsRep1.append(ls[0].item())
        lsRep2.append(ls[1].item())
        lsRep3.append(ls[2].item())

    df = pd.DataFrame(data=list(zip(lsRep1,lsRep2,lsRep3)),columns=["X1","X2","X3"])
    plt = sns.pairplot(df)
    plt.savefig(os.path.join(RESULTSDIR,"assessLatentSpace.png"))    

if __name__=="__main__":
    import sys
    chkptFile = sys.argv[1]
    assessAE(chkptFile)
