'''ItNet Script'''

 #Lambda starts as 1

#Then as itnet trained, UNet refined and lambda varied

#lambde - check LPD initialisation func shows how lambda changed


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

from UNetAlgo1 import UNet
from UNetDataLoader import loadData
import config

img = fdk(A, sinoNoisy)
dev = torch.device("cuda:3")

for i in range(config.ITNET_ITS):                                    # number of iterations
    self.add_module(f"UNet_{i}", UNet().to(dev))
    unet = f"UNet_{i}"
    img = unet(img)
    img = lmda * fdk(A, A*img - sinoNoisy)
    
class ItNet():
    def __init__(self, unet, noIter, lmda, lmdaLearnt, resnetFac):
        super(ItNet, self).__init__()

        self.unet = unet #should be in nn.ModuleList()
        self.noIter = noIter
        self.lmda = nn.ParameterList(
            [nn.Parameter(torch.tensor(lmda[i]), 
            requires_grad=lmdaLearnt[i]) for i in range(len(lmda))])
        self.lmdaLearnt = lmdaLearnt
        self.resnetFac =resnetFac
    
    def forward(self, sino):
        img = fdk(self.A, sino)
        L,D,H,W = img.shape

        s = torch.zeros(L,D,H,W)
        
        for i in range(self.noIter):
            #out = self.unet[i](torch.cat([img, s], dim=1))
            #img = self.resnetFac * img + out[:, 0:1, ...]
            #s = img[:, 1:, ...]
            
            #if i < self.noIter - 1:
            #    img = img - self.lmda[i] * fdk((A, y, img))

            img = self.unet[i](img)
            img = img - self.lmda[i] * fdk(self.A, self.getSino(img) - sino)
        
        return img
    
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
    
    def set_learnable_iteration(self, index):
        for i in list(range(self.get_num_iter_max())):
            if i in index:
                self.lmda[i].requires_grad = True
                self.unet[i].unfreeze()
            else:
                self.lmda[i].requires_grad = False
                self.unet[i].freeze()

    def get_num_iter_max(self):
        return len(self.lmda)
    
    def _print_info(self):
        print("Current lambda(s):")
        print(
            [
                self.lmda[it].item()
                for it in range(len(self.lmda))
                if self.lmda[it].numel() == 1
            ]
        )
        print([self.lmda[it].requires_grad for it in range(len(self.lmda))])
        print("Epoch done", flush=True)