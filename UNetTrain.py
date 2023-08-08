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

from UNet import UNet
from ItNetDataLoader import loadData
import config


dev = torch.device("cuda:3")


noPatients = 0
subDirList = []
cd = "/store/DAMTP/ab2860/AItomotools/data/AItomotools/processed/LIDC-IDRI/"
for subdir in os.listdir(cd):
	f = os.path.join(cd, subdir)
	if not os.path.isfile(f):
		noPatients += 1
		subDirList.append(f)


trainNo = int(np.round(noPatients * 0.8))
validateNo = int(np.round(noPatients * 0.1))
testNo = noPatients - trainNo - validateNo

trainList = subDirList[:trainNo]
validateList = subDirList[trainNo:trainNo+validateNo]
testList = subDirList[trainNo+validateNo:]




trainDS = loadData(imgPaths=trainList, outputSino=False)
testDS = loadData(imgPaths=testList, outputSino=False)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")


trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=False)


testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=False)


unet = UNet().to(dev)
lossFunc = nn.MSELoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE
H = {"train_loss": [], "test_loss": []}


startTime = time.time()
print("Training started at: ", startTime)

valMin = 100

for e in tqdm(range(config.NUM_EPOCHS)):
	unet.train()
	totalTrainLoss = 0
	totalTestLoss = 0
	for (i, (x, y)) in enumerate(trainLoader):
		(x, y) = (x.to(dev), y.to(dev))
		pred = unet(x)

		loss = lossFunc(pred, y)

		opt.zero_grad()
		loss.backward()
		opt.step()
		totalTrainLoss += loss
	with torch.no_grad():
		unet.eval()
		for (x, y) in testLoader:
			(x, y) = (x.to(dev), y.to(dev))
			pred = unet(x)
			totalTestLoss += lossFunc(pred, y)
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	valLoss = avgTestLoss.cpu().detach().numpy()
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))
	if (e+1) % 10 == 0:
		torch.save(
		{'epoch': e+1,
		'model_state_dict': unet.state_dict(),
		'optimizer_state_dict': opt.state_dict(),
		'trainLoss': H["train_loss"],
		'testLoss': H["test_loss"]
		}, 
		"/local/scratch/public/obc22/UNetTrainCheckpts/v2epoch{}.pt".format(e))
	if valLoss < valMin:
		valMin = valLoss
		torch.save(unet.state_dict(),  "/local/scratch/public/obc22/UNetTrainCheckpts/UNetV2MinValSD.pth")

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
plt.savefig("/home/obc22/aitomotools/AItomotools/ItNetTrainLossv2.png")

torch.save(unet, "/local/scratch/public/obc22/UNetTrainCheckpts/UnetV2.pth")
torch.save(unet.state_dict(),  "/local/scratch/public/obc22/UNetTrainCheckpts/UnetV2SD.pth")


print("Done")

#job no 63606

