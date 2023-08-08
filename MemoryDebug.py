import os
import random
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import tomosipo as ts
from ts_algorithms import fdk
import AItomotools.CTtools.ct_utils as ct
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from ItNetDataLoader import loadData
import config


dev = torch.device("cuda:3")

noPatients = 0
subDirList = []
cd = "/local/scratch/public/AItomotools/processed/LIDC-IDRI/"
for subdir in os.listdir(cd):
	f = os.path.join(cd, subdir)
	# checking if it is a file
	if not os.path.isfile(f):
		noPatients += 1
		subDirList.append(f)

trainNo = int(np.round(noPatients * 0.8))
trainList = subDirList[:trainNo]

# define transformations

# create the train and test datasets
trainDS = loadData(imgPaths=trainList, outputSino=True)

# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.ITNET_BATCH_SIZE, pin_memory=False)

for e in tqdm(range(100)):
    for (i, (x, y)) in enumerate(trainLoader):
	    (x, y) = (x.to(dev), y.to(dev))