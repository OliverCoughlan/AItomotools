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

random.shuffle(subDirList)

trainNo = int(np.round(noPatients * 0.8))
validateNo = int(np.round(noPatients * 0.1))
testNo = noPatients - trainNo - validateNo

trainList = subDirList[:trainNo]
validateList = subDirList[trainNo:trainNo+validateNo]
testList = subDirList[trainNo+validateNo:]




# create the train and test datasets
trainDS = loadData(imgPaths=trainList, outputSino=True)
testDS = loadData(imgPaths=testList, outputSino=True)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")


# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=False,
	batch_size=config.ITNET_BATCH_SIZE, pin_memory=False)


testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.ITNET_BATCH_SIZE, pin_memory=False)


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

epochRep = round(config.ITNET_EPOCHS / config.ADAM_REPS)
startEpoch = 0


for optStep in range(config.ADAM_REPS):
	opt = Adam(itnet.parameters())
	print(optStep)

	#tqdm gives a progress bar showing how much training done
	for e in tqdm(range(startEpoch, (optStep+1)*epochRep)):
		startEpoch = (optStep+1)*epochRep 
		# set the model in training mode
		itnet.train()
		# initialize the total traininçg and validation loss
		totalTrainLoss = 0
		totalTestLoss = 0
		# loop over the training set
		for (i, (x, y)) in enumerate(trainLoader):
			# send the input to the device
			(x, y) = (x.to(dev), y.to(dev))
			# perform a forward pass and calculate the training loss

			pred = itnet(x)
			loss = lossFunc(pred, y)
			
			opt.zero_grad()
			loss.backward()
			opt.step()
			# add the loss to the total training loss so far
			totalTrainLoss += loss
		with torch.no_grad():
			itnet.eval()
			# loop over the validation set
			count = 0
			for (x, y) in testLoader:
				# send the input to the device
				(x, y) = (x.to(dev), y.to(dev))
	
				
				count+=1
				pred = itnet(x)
				totalTestLoss += lossFunc(pred, y)
		# calculate the average training and validation loss
		avgTrainLoss = totalTrainLoss / trainSteps
		avgTestLoss = totalTestLoss / testSteps

		H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
		H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

		print("[INFO] EPOCH: {}/{}".format(e + 1, config.ITNET_EPOCHS))
		print("Train loss: {:.6f}, Test loss: {:.4f}".format(
			avgTrainLoss, avgTestLoss))
		if (e+1) % 10 == 0:
			torch.save(
			{'epoch': e+1,
			'model_state_dict': itnet.state_dict(),
			'optimizer_state_dict': opt.state_dict(),
			'trainLoss': H["train_loss"],
			'testLoss': H["test_loss"]
			}, 
			"/local/scratch/public/obc22/ItNetTrainCheckpts/FINALepoch{}.pt".format(e))
	# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))


plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="Training Loss")
plt.plot(H["test_loss"], label="Validation Loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch No.")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("/home/obc22/aitomotools/AItomotools/ItNetTrainLossFINAL.png")

torch.save(itnet, "/local/scratch/public/obc22/ItNetTrainCheckpts/ItNetTrainedFINAL.pth")
torch.save(itnet.state_dict(),  "/local/scratch/public/obc22/ItNetTrainCheckpts/ItNetSDFINAL.pth")

print("Done")

