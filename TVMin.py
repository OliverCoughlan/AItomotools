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

        
