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
    def __init__(self, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
        self.transforms = transforms

        self.dev = torch.device("cuda")
    
        vg = ts.volume(shape=(1, *phantom.shape[1:]), size=(5, 300, 300))
        # Define acquisition geometry. We want fan beam, so lets make a "cone" beam and make it 2D. We also need sod and sdd, so we set them up to something medically reasonable.
        pg = ts.cone(
            angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050
        )
        # A is now an operator.
        self.A = ts.operator(vg, pg)

        

    def getTrainingData(self):
    #this function loops through patient data
    #80% is used for training, 10% for validation, 10% for testing
        
        trainingImgs = []
        for subdir in trainList:
            cd = "/local/scratch/public/AItomotools/processed/LIDC-IDRI/{}".format(subdir)
            for imgs in os.listdir(cd):
                if fnmatch.fnmatch(imgs, 's*'):
                    trainingImgs.append("{}/{}".format(subdir,imgs))
        trainingImgs = trainingImgs[0::40]
        #take every 40th img as this gives c.5000 training imgs
        self.trainingImgs = trainingImgs
        
        return 


    def noTrainImgs(self):
		# return the number of total samples contained in the dataset
        return len(self.trainingImgs)
    
    def loadImgs(self, idx):
		# grab the image path from the current index
        imgToLoad = "/local/scratch/public/AItomotools/processed/LIDC-IDRI/{}".format(self.trainingImgs[idx])
        imgClean = np.load(imgToLoad)
        imgClean = torch.from_numpy(imgClean).to(self.dev) #make in gpu

        sino = self.A(imgClean)

        sino_noisy = ct.sinogram_add_noise(
        sino, I0=10000, sigma=5, crosstalk=0.05, flat_field=None, dark_field=None
        )

        imgNoisy = fdk(A, sino_noisy)
        
        return (imgNoisy, imgClean)
    #good for unet, but later will want noisy sinogram and clean img to train itnet