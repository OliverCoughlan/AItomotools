import os
import random
import numpy as np
import fnmatch
import torch
import torchvision
import tomosipo as ts
from ts_algorithms import tv_min2d
import AItomotools.CTtools.ct_utils as ct
from torch.utils.data import DataLoader


from torch.utils.data import Dataset
from skimage.metrics import structural_similarity as ssim
from ItNetDataLoader import loadData
import config

class tvMin():
    def __init__(self, sino, alpha=0.0001, noIter=100):
        vg = ts.volume(shape=(1, *(512,512)), size=(300/512, 300, 300))
        # Define acquisition geometry. We want fan beam, so lets make a "cone" beam and make it 2D. We also need sod and sdd, so we set them up to something medically reasonable.
        pg = ts.cone(
            angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050
        )
        # A is now an operator.
        self.A = ts.operator(vg, pg)
        self.sino = sino[0]
        self.alpha = alpha
        self.noIter = noIter
        
        
    def forward(self):
        return tv_min2d(self.A, self.sino, self.alpha, self.noIter)

        

""" dev = torch.device("cuda:2")

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

# create the train and test datasets
trainDS = loadData(imgPaths=trainList, outputSino=True)
testDS = loadData(imgPaths=testList, outputSino=True)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")


# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=1, pin_memory=False)

testLoader = DataLoader(testDS, shuffle=False,
	batch_size=1, pin_memory=False)

noise=[]
alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
for j in alphas:
    tmpNoise=[]
    for (i, (x, y)) in enumerate(trainLoader):
        img = tvMin(x, alpha=j)
        img = img.forward()
        img = img.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        y = y[0]
        x = x.cpu().detach().numpy()
        tmpNoise.append(ssim(y[0], img[0], data_range = x[0][0].max()-x[0][0].min()))
    noise.append(np.mean(tmpNoise))

print(noise)

print(np.argmax(noise))
print(alphas[np.argmax(noise)])
 """
#1e-5 found to give best res