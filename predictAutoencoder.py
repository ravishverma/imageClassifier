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

def assessAE(chkptFile,lsSize):
    import pandas as pd
    import seaborn as sns

    predict = predictAE(chkptFile)

    dataset = myDataset(path=DATADIR)

    X = []
    Xnames = ["X"+str(i) for i in range(lsSize)]
    for f in dataset.files:
        ls = predict.reduce(os.path.join(DATADIR,f))
        ls = ls.flatten()
        entry = [lsi.item() for lsi in ls]
        X.append(entry)
        
    df = pd.DataFrame(data=X,columns=Xnames)
    plt = sns.pairplot(df, corner=True, plot_kws={"alpha":0.2})
    plt.savefig(os.path.join(RESULTSDIR,"assessLatentSpace.png"))    

    df["fileName"] = dataset.files
    df.to_csv(os.path.join(RESULTSDIR,"reducedData.csv"))

if __name__=="__main__":
    import sys
    chkptFile = sys.argv[1]
    lsSize = int(sys.argv[2])
    assessAE(chkptFile,lsSize)
