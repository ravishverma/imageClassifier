import torch
from dataLoader import *
from environment import *
from modelLib import *

class predictAE:
    def __init__(self, chkptFile):
        self.model = autoencoder().to(DEVICE)
        self.model, _, _, _, _ = loadChkPt(chkptFile, self.model) 
        self.dataset = myDataset()

    def reduce(self, filepath):
        img = self.dataset.loadFromFile(filepath)
        img = img.to(DEVICE)

        self.model.eval()

        with torch.no_grad():
            ls, _ = self.model(img)
            return ls.detach().numpy()

def assessAE(chkptFile):
    import pandas as pd
    import seaborn as sns

    predict = predictAE(chkptFile)

    dataset = myDataset(path=DATADIR)

    X1 = []
    X2 = []
    X3 = []
    X4 = []
    for f in dataset.files:
        ls = predict.reduce(os.path.join(DATADIR,f))
        X1.append(ls[0].item())
        X2.append(ls[1].item())
        X3.append(ls[2].item())
        X4.append(ls[3].item())

    df = pd.DataFrame(data=list(zip(X1,X2,X3,X4)),columns=["X1","X2","X3","X4"])
    plt = sns.pairplot(df, plot_kws={"alpha":0.2})
    plt.savefig(os.path.join(RESULTSDIR,"assessLatentSpace.png"))    

if __name__=="__main__":
    import sys
    chkptFile = sys.argv[1]
    assessAE(chkptFile)
