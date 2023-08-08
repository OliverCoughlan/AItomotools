import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import config
import AItomotools.CTtools.ct_utils as ct
import tomosipo as ts
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from ts_algorithms import fdk
from ItNetDataLoader import loadData

from AItomotoolsNEW.models.ItNet import ItNet, UNet

dev = torch.device("cuda:2")
unet = UNet.load('/store/DAMTP/ab2860/wip_models/UNet_final_iter.pt')[1]# iterate over the randomly selected test image paths
print(unet)