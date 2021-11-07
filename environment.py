import torch
DATADIR = "./data"
RESULTSDIR = "./results"
BATCHSIZE = 4
EPOCH0 = 0
NEPOCHS = 50
SAVEINTERVAL = 10
RANDOMSEED = 42
DATALIMIT = None
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
