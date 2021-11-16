import pandas as pd
from environment import *
import os
import seaborn as sns
import sys

df = pd.read_csv(os.path.join(RESULTSDIR,"reducedData.csv"), index_col=0)

def clusterWithAC():
    from sklearn.cluster import AgglomerativeClustering
    ac = AgglomerativeClustering()
    clustering = ac.fit(X=df[Xnames].to_numpy())
    return clustering.labels_

def clusterWithKM():
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3).fit(X=df[Xnames].to_numpy())
    return km.labels_

if __name__=="__main__":
    labels = None

    lsSize = int(sys.argv[1])
    algo = sys.argv[2]
    Xnames = ["X"+str(i) for i in range(lsSize)]

    if algo=="AC":
        labels = clusterWithAC()
    elif algo=="KM":
        labels = clusterWithKM()

    df["labels"] = [str(i) for i in labels]

    df.to_csv(os.path.join(RESULTSDIR,"labeledData.csv"))
    plt = sns.pairplot(df,corner=True,hue="labels",plot_kws={"alpha":0.2})
    plt.savefig(os.path.join(RESULTSDIR,"clustersLatentSpace.png"))
