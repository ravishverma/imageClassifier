import pandas as pd
from environment import *
import os
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import sys

lsSize = int(sys.argv[1])
Xnames = ["X"+str(i) for i in range(lsSize)]

df = pd.read_csv(os.path.join(RESULTSDIR,"reducedData.csv"), index_col=0)

ac = AgglomerativeClustering()
clustering = ac.fit(X=df[Xnames].to_numpy())

df["labels"] = [str(i) for i in clustering.labels_]

df.to_csv(os.path.join(RESULTSDIR,"labeledData.csv"))

plt = sns.pairplot(df,corner=True,hue="labels",plot_kws={"alpha":0.2})
plt.savefig(os.path.join(RESULTSDIR,"clustersLatentSpace.png"))
