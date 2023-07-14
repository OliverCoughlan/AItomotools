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

from ItNet import ItNet
from ItNetDataLoader import loadData
import config


dev = torch.device("cuda:1")


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
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])

# create the train and test datasets
trainDS = loadData(transforms=transforms, imgPaths=trainList, outputSino=True)
testDS = loadData(transforms=transforms, imgPaths=testList, outputSino=True)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")


# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.ITNET_BATCH_SIZE, pin_memory=True)


testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.ITNET_BATCH_SIZE, pin_memory=True)


# initialize our UNet model
itnet = ItNet().to(dev)
# initialize loss function and optimizer
lossFunc = nn.MSELoss()
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.ITNET_BATCH_SIZE
testSteps = len(testDS) // config.ITNET_BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}


# loop over epochs
startTime = time.time()
print("Training started at: ", startTime)

checkpoint = torch.load("/local/scratch/public/obc22/trainCheckpts/.pt")
itnet.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

epochRep = round(config.ITNET_EPOCHS / config.ADAM_REPS)
startRep=epoch//epochRep
startEpoch=epoch%epochRep + epochRep*startRep

for optStep in range(startRep, config.ADAM_REPS):
	opt = Adam(itnet.parameters())
	if optStep == startRep:
	    opt.load_state_dict(checkpoint['optimizer_state_dict'])
	#tqdm gives a progress bar showing how much training done
	for e in tqdm(range(startEpoch, (optStep+1)*epochRep)):
		startEpoch = (optStep+1)*epochRep # for later, in case i>2
		# set the model in training mode
		itnet.train()
		# initialize the total traininçg and validation loss
		totalTrainLoss = 0
		totalTestLoss = 0
		# loop over the training set
		for (i, (x, y)) in enumerate(trainLoader):
			#print(i)
			# send the input to the device
			(x, y) = (x.to(dev), y.to(dev))
			# perform a forward pass and calculate the training loss
				

			pred = itnet(x)
			#pred = pred / torch.max(pred)
			#y = y / torch.max(y)
			loss = lossFunc(pred, y)
			# first, zero out any previously accumulated gradients, then
			# perform backpropagation, and then update model parameters
			opt.zero_grad()
			loss.backward()
			opt.step()
			# add the loss to the total training loss so far
			totalTrainLoss += loss
		# switch off autograd
		with torch.no_grad():
			# set the model in evaluation modde
			itnet.eval()
			# loop over the validation set
			count = 0
			for (x, y) in testLoader:
				# send the input to the device
				(x, y) = (x.to(dev), y.to(dev))
				# make the predictions and calculate the validation loss
				
				#print(count)
				count+=1
				pred = itnet(x)
				totalTestLoss += lossFunc(pred, y)
		# calculate the average training and validation loss
		avgTrainLoss = totalTrainLoss / trainSteps
		avgTestLoss = totalTestLoss / testSteps
		# update our training history
		H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
		H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, config.ITNET_EPOCHS))
		print("Train loss: {:.6f}, Test loss: {:.4f}".format(
			avgTrainLoss, avgTestLoss))
		if (e+1) % 40 == 0:
			torch.save(
			{'epoch': e+1,
			'model_state_dict': unet.state_dict(),
			'optimizer_state_dict': opt.state_dict(),
			'loss': loss,
			}, 
			"/local/scratch/public/obc22/trainCheckpts/ItNetepoch{}.pt".format(e))
	# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
#plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("/home/obc22/aitomotools/AItomotools/ITNETtmp.png")

torch.save(itnet, "/local/scratch/public/obc22/trainCheckpts/trainedITNET.pth")


print("Done")

#job no 63606