import os
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
import torch
import torchvision
import tomosipo as ts
from ts_algorithms import fdk
import AItomotools.CTtools.ct_utils as ct
from torch.utils.data import Dataset
import AItomotools.CTtools.ct_utils as ct
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time


class loadData(Dataset):
    def __init__(self, transforms, imgPaths):

        self.transforms = transforms
        #transforms to be used
        self.trainList = imgPaths
        #img files to be trained on
        self.dev = torch.device("cuda:0")
    
        trainingImgs = []
        #list of complete paths to training imgs
        for subdir in self.trainList:
            for imgs in os.listdir(subdir):
                if fnmatch.fnmatch(imgs, 's*'):
                    #checks img is a slice
                    trainingImgs.append(os.path.join(subdir, imgs))
                    #adds the img to the list of training paths
                 
        trainingImgs = trainingImgs[0::40] 
        #take every 40th img as this gives c.5000 training imgs
        self.trainingImgs = trainingImgs


    def getSino(self, imgClean):
        #takes clean img and turns into a sinogram
        imgClean = np.expand_dims(imgClean, 0).astype("float32")
        vg = ts.volume(shape=(1, *imgClean.shape[1:]), size=(5, 300, 300))
        # Define acquisition geometry. We want fan beam, so lets make a "cone" beam and make it 2D. We also need sod and sdd, so we set them up to something medically reasonable.
        pg = ts.cone(
            angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050
        )
        # A is now an operator.
        self.A = ts.operator(vg, pg)
        return self.A(imgClean)


    def __len__(self):
		# return the number of total samples contained in the dataset
        return len(self.trainingImgs)
    
    def __getitem__(self, idx):
		# grab the image path from the current index
        imgToLoad = self.trainingImgs[idx]
        imgClean = np.load(imgToLoad)
        imgClean = ct.from_HU_to_mu(imgClean)
        imgClean = torch.from_numpy(imgClean)#.to(self.dev) #make in gpu

        sino = self.getSino(imgClean)

        sino_noisy = ct.sinogram_add_noise(
        sino, I0=10000, sigma=5, crosstalk=0.05, flat_field=None, dark_field=None
        )

        imgNoisy = fdk(self.A, sino_noisy)
        
        return (imgNoisy, imgClean)
    #good for unet, but later will want noisy sinogram and clean img to train itnet

transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((512,
		512)),
	transforms.ToTensor()])

noPatients = 0
subDirList = []
cd = "/local/scratch/public/AItomotools/processed/LIDC-IDRI/"
for file in os.listdir(cd):
	f = os.path.join(cd, file)
	# checking if it is a file
	if os.path.isfile(f) != True:
		noPatients += 1
		subDirList.append(f)
trainNo = int(np.round(noPatients * 0.8))
trainList = subDirList[:trainNo]


trainDS = loadData(transforms=transforms, imgPaths=trainList)

trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=10, pin_memory=True) #10 used here as a temporary amount for min rep ex.

for (i, (x,y)) in enumerate(trainLoader):
	print(i)