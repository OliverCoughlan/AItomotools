'''Model Eval'''

#Compare UNet (pure ML), TS algo - tv  (pure itertive) and ItNet (balance)
#ItNet should be best - interesting narrative

#take ImgClean and output from ItNet and UNet and 
#avg rms error
# PSNR
# do for each img, get vals, plot dist, find mean, sd (if gaussian) etc

#violin plots  for error


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
import time

from UNet import UNet
from ItNetDataLoader import loadData
import config


dev = torch.device("cuda:3")


#split up training and testing imgs
noPatients = 0
subDirList = []
cd = "/local/scratch/public/AItomotools/processed/LIDC-IDRI/"
for subdir in os.listdir(cd):
	f = os.path.join(cd, subdir)
	# checking if it is a file
	if not os.path.isfile(f):
		noPatients += 1
		subDirList.append(f)

#random.shuffle(subDirList)

trainNo = int(np.round(noPatients * 0.8))
validateNo = int(np.round(noPatients * 0.1))
testNo = noPatients - trainNo - validateNo

trainList = subDirList[:trainNo]
validateList = subDirList[trainNo:trainNo+validateNo]
testList = subDirList[trainNo+validateNo:]



# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])

# create the train and test datasets
testDS = loadData(transforms=transforms, imgPaths=testList, outputSino=False)
print(f"[INFO] found {len(testDS)} examples in the test set...")


# create the training and test data loaders

testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=True)


# initialize our UNet model
unet = UNet().to(dev)
itnet = ItNet().to(dev)

for (i, (x, y)) in enumerate(testLoader):
		# send the input to the device
		(x, y) = (x.to(dev), y.to(dev))
		# perform a forward pass and calculate the training loss
		unetPred = unet(x)
		itnetPred = itnet(x)
		tsPred = ts.tv(x)
		
		unetErr = []
		itnetErr = []
		tsErr = []
		unetErr.append(PSNR(unetPred, y))
		itnetErr.append(PSNR(itnetPred, y))
		tsErr.append(PSNR(tsPred, y))




		
	