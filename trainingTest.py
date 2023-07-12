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

dev = torch.device("cuda:1")




def plots(noisyImg, cleanImg, predImg):
	# initialize our figure

	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

	a = ax[0].imshow(noisyImg[0].T, vmin=0,vmax=2.5)
	b = ax[1].imshow(cleanImg[0].T, vmin=0,vmax=2.5)
	c = ax[2].imshow(predImg[0].T, vmin=0,vmax=2.5)
	# set the titles of the subplots
	ax[0].set_title("Noisy Image")
	ax[1].set_title("Clean Image")
	ax[2].set_title("Predicted Image")
	# set the layout of the figure and display it
	figure.subplots_adjust(right=0.8)
	cbar_ax = figure.add_axes([0.15, 0.3, 0.75, 0.02])
	figure.colorbar(c, cax=cbar_ax, location = "bottom")
	figure.tight_layout()
	plt.savefig("/home/obc22/aitomotools/AItomotools/imgComparison.png")
	

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	from torchvision import transforms
	transforms = transforms.Compose([transforms.ToPILImage(),
			transforms.Resize((config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_WIDTH)),
			transforms.ToTensor()])
	
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

		trainDS = loadData(transforms=transforms, imgPaths=trainList)
		trainLoader = DataLoader(trainDS, shuffle=True,
		batch_size=config.BATCH_SIZE, pin_memory=True)
		
		for (i, (x, y)) in enumerate(trainLoader):
				x = x.to(dev)
				y = y.to(dev)
				

		noisyImg = x
		cleanImg = y

		predImg = model(x).squeeze()
		predImg = predImg.cpu().numpy()
		cleanImg = cleanImg.cpu().numpy()
		noisyImg = noisyImg.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		#predImg = (predImg > config.THRESHOLD) * 255
		#predImg = predImg.astype(np.uint8)
		# prepare a plot for visualization
		plots(noisyImg, cleanImg, predImg)


		return

print("[INFO] loading up test image paths...")
#imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
#imagePaths = np.random.choice(imagePaths, size=10)
imagePaths = ["/local/scratch/public/AItomotools/processed/LIDC-IDRI/LIDC-IDRI-0001/slice_0.npy"]

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load("/home/obc22/aitomotools/AItomotools/trained.pth").to(dev)
# iterate over the randomly selected test image paths
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, path)