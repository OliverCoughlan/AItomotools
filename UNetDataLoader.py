import os
import random
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
import torch
import torchvision
import tomosipo as ts
from ts_algorithms import fdk
import AItomotools.CTtools.ct_utils as ct

from torch.utils.data import Dataset

class loadData(Dataset):
    def __init__(self, transforms, imgPaths, outputSino):
		# store the image and mask filepaths, and augmentation
		# transforms
        self.transforms = transforms
        self.trainList = imgPaths
        self.dev = torch.device("cuda:3")
        self.outputSino = outputSino
    
        trainingImgs = []
        #list of complete paths to training imgs
        for subdir in self.trainList:
            for imgs in os.listdir(subdir):
                if fnmatch.fnmatch(imgs, 's*'):
                    #checks img is a slice
                    trainingImgs.append(os.path.join(subdir, imgs))
                    #adds the img to the list of training paths
        trainingImgs = trainingImgs[0::47] 
        if len(trainingImgs) > 4000:
            trainingImgs = trainingImgs[0:4000]
        #take every 40th img as this gives c.5000 training imgs
        self.trainingImgs = trainingImgs
        print(trainingImgs[0])

    def getSino(self, imgClean):
        #takes clean img and turns into a sinogram
        #vg = ts.volume(shape=(1, *imgClean.shape[1:]), size=(5, 300, 300))
        vg = ts.volume(shape=(1, *imgClean.shape[1:]), size=(300/imgClean.shape[1], 300, 300))
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
        imgClean[imgClean < -1000] = -1000
        imgClean = ct.from_HU_to_mu(imgClean)
        imgClean = torch.from_numpy(imgClean)#.to(self.dev) #make in gpu
        imgClean = np.expand_dims(imgClean, 0).astype("float32")


        sino = self.getSino(imgClean)

        sino_noisy = ct.sinogram_add_noise(
        sino, I0=3500, sigma=5, crosstalk=0.05, flat_field=None, dark_field=None
        )
        sino_noisy = torch.from_numpy(sino_noisy)
        output = sino_noisy.type(torch.float32)

        if not self.outputSino:
            output = fdk(self.A, output)
        return (output, imgClean)
    #good for unet, but later will want noisy sinogram and clean img to train itnet