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
from UNet import UNet as UNet2

dev = torch.device("cuda:2")



def plots(noisyImg, cleanImg, predImg,i2):
	# initialize our figure

	figure, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 10))


	a = ax[0].imshow(noisyImg[0].T, vmin=0,vmax=2.5)
	b = ax[1].imshow(cleanImg[0].T, vmin=0,vmax=2.5)
	c = ax[2].imshow(predImg[0].T, vmin=0,vmax=2.5)
	d = ax[3].imshow(i2[0].T, vmin=0,vmax=2.5)
	# set the titles of the subplots
	ax[0].set_title("Noisy Image")
	ax[1].set_title("Clean Image")
	ax[2].set_title("Ander's UNet")
	ax[3].set_title("Oliver's UNet")
	# set the layout of the figure and display it
	figure.subplots_adjust(right=0.8)
	cbar_ax = figure.add_axes([0.15, 0.3, 0.75, 0.02])
	figure.colorbar(c, cax=cbar_ax, location = "bottom")
	figure.tight_layout()
	plt.savefig("/home/obc22/aitomotools/AItomotools/imgComparison.png")
	

def make_predictions(model,m2, imagePath):
	# set model to evaluation mode
	model.eval()
	m2.eval()
	# turn off gradient tracking

	
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		
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
		testList = subDirList[trainNo:]

		trainDS = loadData(imgPaths=trainList, outputSino=False)
		trainLoader = DataLoader(trainDS, shuffle=True,
		batch_size=config.BATCH_SIZE, pin_memory=False)

		dev = torch.device("cuda:2")
		
		for (i, (x, y)) in enumerate(trainLoader):
			if i < 1:
				print(i)
				x = x.to(dev)
				y = y.to(dev)
			else:
				break

		x = x.to(dev)
		y = y.to(dev)

		noisyImg = x
		cleanImg = y

		predImg = model(x).squeeze(1)
		predImg = predImg.cpu().numpy()
		i2 = m2(x).squeeze(1).cpu().numpy()
		cleanImg = cleanImg.cpu().numpy()
		noisyImg = noisyImg.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		#predImg = (predImg > config.THRESHOLD) * 255
		#predImg = predImg.astype(np.uint8)
		# prepare a plot for visualization
		plots(noisyImg, cleanImg, predImg, i2)


		return

print("[INFO] loading up test image paths...")
#imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
#imagePaths = np.random.choice(imagePaths, size=10)
imagePaths = ["/local/scratch/public/AItomotools/processed/LIDC-IDRI/LIDC-IDRI-0001/slice_0.npy"]

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
#unet = torch.load("/local/scratch/public/obc22/trainCheckpts/trainedUNET.pth").to(dev)
unet = UNet.load('/store/DAMTP/ab2860/wip_models/UNet_final_iter.pt')[0].to(dev)# iterate over the randomly selected test image paths

u2 = UNet2().to(dev)
unetSD = torch.load("/local/scratch/public/obc22/UNetTrainCheckpts/trainedUNETstatedict.pth")
u2.load_state_dict(unetSD)
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, u2, path)