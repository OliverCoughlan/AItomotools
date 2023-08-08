'''Model Eval'''

#Compare UNet (pure ML), TS algo - tv  (pure itertive) and ItNet (balance)
#ItNet should be best - interesting narrative

#take ImgClean and output from ItNet and UNet and 
#avg rms error
# PSNR
# do for each img, get vals, plot dist, find mean, sd (if gaussian) etc

#violin plots  for error

#also add fdk into comparison as baseline with noise
#likewise for ground truth image

#tv min rec_tv_min = tv_min2d(A, y, 0.0001, num_iterations=100)

#0.0001 is alpha
#alpha should be proportional to noise in image
#run SSIM on subset of images and use trial an error to pick alpha
#for i in range 5:
#go through 5 images
#change alpha several times (chanbge order of mag)
#pick best one
#also try fdk as tv should look better - if not something is wrong (alpha)

#add table of results - different noise metrics, runtime for eval (run 100 and div by 100)
#mention training time somewhere but not in final table
#add zoom plots

import os
import random
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import tomosipo as ts
#import pandas as pd
from ts_algorithms import fdk
import AItomotools.CTtools.ct_utils as ct
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time

from UNet import UNet
from ItNet import ItNet
from TVMin import tvMin
from ItNetDataLoader import loadData
import config

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

dev = torch.device("cuda:2")

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



# create the train and test datasets
testDS = loadData(imgPaths=testList, outputSino=True)
testDSimg = loadData(imgPaths=testList, outputSino=False)
print(f"[INFO] found {len(testDS)} examples in the test set...")

# create the training and test data loaders

testLoaderFDK = DataLoader(testDS, shuffle=False,
	batch_size=1, pin_memory=False)
testLoaderUNet = DataLoader(testDSimg, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=False)
testLoaderItNet = DataLoader(testDS, shuffle=False,
	batch_size=config.ITNET_BATCH_SIZE, pin_memory=False)
testLoaderTV = DataLoader(testDS, shuffle=False,
	batch_size=1, pin_memory=False)


# initialize our UNet model
unet = UNet().to(dev)
unetSD = torch.load("/local/scratch/public/obc22/UNetTrainCheckpts/trainedUNETstatedict.pth")
unet.load_state_dict(unetSD)
itnet = ItNet().to(dev)
itnetSD = torch.load("/local/scratch/public/obc22/ItNetTrainCheckpts/FINALepoch209.pt")['model_state_dict']
itnet.load_state_dict(itnetSD)


unet.eval()
itnet.eval()

unetErr = np.zeros((len(testLoaderUNet), 3))
itnetErr = np.zeros((len(testLoaderItNet), 3))
tvErr = np.zeros((len(testLoaderTV), 3))
fdkErr = np.zeros((len(testLoaderFDK), 3))

vg = ts.volume(shape=(1, *(512,512)), size=(300/512, 300, 300))
# Define acquisition geometry. We want fan beam, so lets make a "cone" beam and make it 2D. We also need sod and sdd, so we set them up to something medically reasonable.
pg = ts.cone(
	angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050
)
# A is now an operator.
A = ts.operator(vg, pg)

fdkStart = time.time()
for (i, (x, y)) in enumerate(testLoaderFDK):
		(x, y) = (x[0].to(dev), y[0].to(dev))
		fdkPred = fdk(A, x)
		
		fdkPred = fdkPred[0].cpu().detach().numpy()
		y = y[0].cpu().detach().numpy()
		#print("y")
		#print(y.shape)
		#print("fdk")
		#print(fdkPred.shape)
		
		SSIMfdk = ssim(y, fdkPred, data_range = fdkPred.max()-fdkPred.min())
		MSEfdk = mse(y, fdkPred)
		PSNRfdk = psnr(y, fdkPred, data_range = fdkPred.max()-fdkPred.min())

		fdkErr[i,:] = [SSIMfdk, MSEfdk, PSNRfdk]
fdkEnd = time.time() - fdkStart
print(fdkEnd)
print(fdkEnd/len(testDS))

unetStart = time.time()
uSSIMmin = 100
for (i, (x, y)) in enumerate(testLoaderUNet):
		(x, y) = (x.to(dev), y[0].to(dev))
		
		unetPred = unet(x).squeeze(1)

		unetPred = unetPred.cpu().detach().numpy()
		y = y[0].cpu().detach().numpy().squeeze()
		unetPred = unetPred[0]

		SSIMUnet = ssim(y, unetPred, data_range = unetPred.max()-unetPred.min())
		MSEUnet = mse(y, unetPred)
		PSNRUnet = psnr(y, unetPred, data_range = unetPred.max()-unetPred.min())
		
		unetErr[i,:] = [SSIMUnet, MSEUnet, PSNRUnet]
		
		#if SSIMUnet < uSSIMmin:
		#	uSSIMmin = SSIMUnet
		#	uMinIdx = i

unetEnd = time.time() - unetStart
print(unetEnd)
print(unetEnd/len(testDS))
"""
for (i, (x, y)) in enumerate(testLoaderUNet):
		if i == uMinIdx:
			(x, y) = (x.to(dev), y[0].to(dev))
			
			fW=x[0].cpu().detach().numpy()
			uW = unet(x).squeeze(1)
			vg = ts.volume(shape=(1, *x[0].shape[1:]), size=(300/x[0].shape[1], 300, 300))
			pg = ts.cone(angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050)
			A = ts.operator(vg, pg)
			j = torch.zeros(config.BATCH_SIZE,360,900)
			for i in range(j.shape[0]):
				j[i] = A(x[i])
			x = j
			x = torch.unsqueeze(x,1)
			iW = itnet(x)[0].cpu().detach().numpy()
			tw = tvMin(x).forward()[0].cpu().detach().numpy()
			y = y[0].cpu().detach().numpy()
			
			figure, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 10))

			b = ax[0,1].imshow(y[0].T, vmin=0,vmax=2.5)
			a = ax[0,0].imshow(fW[0].T, vmin=0,vmax=2.5)
			c = ax[0,2].imshow(uW.T, vmin=0,vmax=2.5)
			# set the titles of the subplots
			e = ax[1,1].imshow(iW.T, vmin=0,vmax=2.5)
			d = ax[1,0].imshow(tw.T, vmin=0,vmax=2.5)
			f = ax[1,2].imshow(tw.T, vmin=0,vmax=2.5)
			
			plt.savefig("/home/obc22/aitomotools/AItomotools/tmptmp.png")
"""


itNetStart = time.time()
for (i, (x, y)) in enumerate(testLoaderItNet):
		(x, y) = (x.to(dev), y[0].to(dev))
		
		itnetPred = itnet(x)

		itnetPred = itnetPred[0].cpu().detach().numpy()
		y = y[0].cpu().detach().numpy()

		itnetPred = itnetPred[0]

		SSIMItNet = ssim(y, itnetPred, data_range = itnetPred.max()-itnetPred.min())
		MSEItNet = mse(y, itnetPred)
		PSNRItNet = psnr(y, itnetPred, data_range = itnetPred.max()-itnetPred.min())

		itnetErr[i,:] = [SSIMItNet, MSEItNet, PSNRItNet]
itNetEnd = time.time() - itNetStart
print(itNetEnd)
print(itNetEnd/len(testDS))

tvStart = time.time()
for (i, (x, y)) in enumerate(testLoaderTV):
		(x, y) = (x[0].to(dev), y[0].to(dev))
		tvPred = tvMin(x).forward()

		tvPred = tvPred[0].cpu().detach().numpy()
		y = y[0].cpu().detach().numpy()
		
		SSIMTVmin = ssim(y, tvPred, data_range = tvPred.max()-tvPred.min())
		MSETVmin = mse(y, tvPred)
		PSNRTVmin = psnr(y, tvPred, data_range = tvPred.max()-tvPred.min())

		tvErr[i,:] = [SSIMTVmin, MSETVmin, PSNRTVmin]
tvEnd = time.time() - tvStart
print(tvEnd)
print(tvEnd/len(testDS))

unetBest = np.argmax(unetErr[:,0])
unetWorst = np.argmin(unetErr[:,0])
itnetBest = np.argmax(itnetErr[:,0])
itnetWorst = np.argmin(itnetErr[:,0])
tvBest = np.argmax(tvErr[:,0])
tvWorst = np.argmin(tvErr[:,0])

#from ModelReconComparison import compPred, worstComp
print(unetErr[:,0])
print(unetBest)
print(unetWorst)
print(tvErr[:,0])
print(tvBest)
print(tvWorst)
print(itnetErr[:,0])


print(itnetBest)
print(itnetWorst)

#compPred(unet, itnet, "UNet", unetBest, unetWorst)
#compPred(unet, itnet, "TV", tvBest, tvWorst)
#compPred(unet, itnet, "ItNet", itnetBest, itnetWorst)

#worstComp(unet, itnet, itnetWorst)

print("FDK:  SSIM = ",np.mean(fdkErr[:,0]), ", MSE = ",np.mean(fdkErr[:,1]), ", PSNR = ",np.mean(fdkErr[:,2]))
print("UNET:  SSIM = ",np.mean(unetErr[:,0]), ", MSE = ",np.mean(unetErr[:,1]), ", PSNR = ",np.mean(unetErr[:,2]))
print("ITNET:  SSIM = ",np.mean(itnetErr[:,0]), ", MSE = ",np.mean(itnetErr[:,1]), ", PSNR = ",np.mean(itnetErr[:,2]))
print("TVMIN:  SSIM = ",np.mean(tvErr[:,0]), ", MSE = ",np.mean(tvErr[:,1]), ", PSNR = ",np.mean(tvErr[:,2]))


figure, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))
plt.ylim([0,1])

a = ax[0].violinplot(fdkErr[:,0])
b = ax[1].violinplot(tvErr[:,0])
c = ax[2].violinplot(unetErr[:,0])
d = ax[3].violinplot(itnetErr[:,0])
# set the titles of the subplots
ax[0].set_title("FDK")
ax[1].set_title("TV")
ax[2].set_title("UNet")
ax[3].set_title("ItNet")
# set the layout of the figure and display it
figure.subplots_adjust(right=0.8)
figure.tight_layout()
ax[0].set_ylim([0,1])
ax[1].set_ylim([0,1])
ax[2].set_ylim([0,1])
ax[3].set_ylim([0,1])
ax[0].get_xaxis().set_visible(False)
ax[1].get_xaxis().set_visible(False)
ax[2].get_xaxis().set_visible(False)
ax[3].get_xaxis().set_visible(False)
plt.savefig("/home/obc22/aitomotools/AItomotools/SSIMErr.png")
	
figure, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))
plt.ylim([0,1])

a = ax[0].violinplot(fdkErr[:,1])
b = ax[1].violinplot(tvErr[:,1])
c = ax[2].violinplot(unetErr[:,1])
d = ax[3].violinplot(itnetErr[:,1])
# set the titles of the subplots
ax[0].set_title("FDK")
ax[1].set_title("TV")
ax[2].set_title("UNet")
ax[3].set_title("ItNet")
# set the layout of the figure and display it
figure.subplots_adjust(right=0.8)
figure.tight_layout()
ax[0].set_ylim([0,np.amax(fdkErr[:,1])+.1])
ax[1].set_ylim([0,np.amax(fdkErr[:,1])+.1])
ax[2].set_ylim([0,np.amax(fdkErr[:,1])+.1])
ax[3].set_ylim([0,np.amax(fdkErr[:,1])+.1])
ax[0].get_xaxis().set_visible(False)
ax[1].get_xaxis().set_visible(False)
ax[2].get_xaxis().set_visible(False)
ax[3].get_xaxis().set_visible(False)
plt.savefig("/home/obc22/aitomotools/AItomotools/MSEErr.png")
	
figure, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))
plt.ylim([0,1])

a = ax[0].violinplot(fdkErr[:,2])
b = ax[1].violinplot(tvErr[:,2])
c = ax[2].violinplot(unetErr[:,2])
d = ax[3].violinplot(itnetErr[:,2])
# set the titles of the subplots
ax[0].set_title("FDK")
ax[1].set_title("TV")
ax[2].set_title("UNet")
ax[3].set_title("ItNet")
# set the layout of the figure and display it
figure.subplots_adjust(right=0.8)
figure.tight_layout()
ax[0].set_ylim([np.amin(tvErr[:,2]),np.amax(itnetErr[:,2])+1])
ax[1].set_ylim([np.amin(tvErr[:,2]),np.amax(itnetErr[:,2])+1])
ax[2].set_ylim([np.amin(tvErr[:,2]),np.amax(itnetErr[:,2])+1])
ax[3].set_ylim([np.amin(tvErr[:,2]),np.amax(itnetErr[:,2])+1])
ax[0].get_xaxis().set_visible(False)
ax[1].get_xaxis().set_visible(False)
ax[2].get_xaxis().set_visible(False)
ax[3].get_xaxis().set_visible(False)
plt.savefig("/home/obc22/aitomotools/AItomotools/PSNRErr.png")
	
""" 

for (i, (x, y)) in enumerate(testLoader):
		(x, y) = (x.to(dev), y.to(dev))
		fdkPred = fdk(A, x)
		unetPred = unet(fdkPred)
		itnetPred = itnet(x)
		tvPred = tvMin(x).forward()
		
		SSIMfdk = ssim(y, fdkPred, data_range = fdkPred.max()-fdkPred.min())
		SSIMUnet = ssim(y, unetPred, data_range = unetPred.max()-unetPred.min())
		SSIMItNet = ssim(y, itnetPred, data_range = itnetPred.max()-itnetPred.min())
		SSIMTVmin = ssim(y, tvPred, data_range = tvPred.max()-tvPred.min())

		MSEfdk = mse(y, fdkPred)
		MSEUnet = mse(y, unetPred)
		MSEItNet = mse(y, itnetPred)
		MSETVmin = mse(y, tvPred)
		
		PSNRfdk = psnr(y, fdkPred)
		PSNRUnet = psnr(y, unetPred)
		PSNRItNet = psnr(y, itnetPred)
		PSNRTVmin = psnr(y, tvPred)
		
		fdkErr[i,:] = [SSIMfdk, MSEfdk, PSNRfdk]
		unetErr[i,:] = [SSIMUnet, MSEUnet, PSNRUnet]
		itnetErr[i,:] = [SSIMItNet, MSEItNet, PSNRItNet]
		tvErr[i,:] = [SSIMTVmin, MSETVmin, PSNRTVmin] """