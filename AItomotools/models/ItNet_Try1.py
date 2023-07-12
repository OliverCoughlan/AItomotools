from AItomotools.utils.math import power_method
from AItomotools.utils.parameter import Parameter
import AItomotools.CTtools.ct_geometry as ct

import numpy as np

import tomosipo as ts
from tomosipo.torch_support import to_autograd
from ts_algorithms import fbp #importing conventional fdk which is used in lpd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop


##Code for UNet
class Block(nn.Module):
	#class takes input map, applies 2 ReLu convolutions and then gives output map
	def __init__(self, inChannels, outChannels):
		super().__init__()
		#creates two convolution layers and one relu layer
		self.conv1 = nn.Conv2d(inChannels, outChannels, 3)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(outChannels, outChannels, 3)
	
	def forward(self, x):
		#executes the above 3 layers to the input map to get the output
		print(self.conv1(x))
		return self.conv2(self.relu(self.conv1(x)))
    

class Encoder(nn.Module):
	#encoder part of the unet algorithm
	def __init__(self, channels=(3, 16, 32, 64, 128, 256)):
		#channels is no. of channels in each layer, starting with input layer
		super().__init__()
		#makes super class so can be called by other classes
		self.encBlocks = nn.ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		#initialises a list of block objects (from class above) which constitute the different blocks in the network
		self.pool = nn.MaxPool2d(2)
		#reduces the dim of subsequent maps by factor of 2, akin to going down the U in the Unet

	def forward(self, x):
		#runs the encoder stage of Unet
		blockOutputs = []
		#loops thru all encoder blocks, processing inputs and storing outputs.  Then reduces dim by 2x
		for block in self.encBlocks:
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		return blockOutputs


class Decoder(nn.Module):
	#decoder is last part - we now go up the U
	def __init__(self, channels=(256, 128, 64, 32, 16)):
		#now going the other way, so no. of channels decreases
		super().__init__()
		#again, super class
		self.channels = channels
		self.upconvs = nn.ModuleList(
			[nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		#now initialise decoder blocks, with less channels but more dims as you go 
		self.dec_blocks = nn.ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		
	def forward(self, x, encFeatures):
		#loop through the channels and decode image
		for i in range(len(self.channels) - 1):
			x = self.upconvs[i](x)
			#images goes thriough upsampling blocks
			encFeat = self.crop(encFeatures[i], x)
			#crop encoded features and run through decoder
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		return x
        #returns final output
	
	def crop(self, encFeatures, x):
		#crops encoder images to match dims of decoder images
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		return encFeatures
	

class UNet(nn.Module):
	def __init__(self, 
	     encChannels=(3, 16, 32, 64, 128, 256),
	     #encoder channels - increasing
		 decChannels=(256, 128, 64, 32, 16),
		 #decoder chanbnels - decreasing
		 nbClasses=1, 
		 #no. of segemntation classes used to clasify each pixel
	     retainDim=True,
	     #should output img be same dim as original?
		 outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
		 #dim of output img

		super().__init__()
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		#call encoder and decoder
		self.head = nn.Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize
		
	def forward(self, x):
		#runs Unet all the way through
		encFeatures = self.encoder(x)
		#runs the encoder through to get key features
		decFeatures = self.decoder(encFeatures[::-1][0],
			encFeatures[::-1][1:])
		#runs these features back through the decoder and checks dims are right
		map = self.head(decFeatures)
		#runs thru regression to get segmentation mask
		if self.retainDim:
			map = F.interpolate(map, self.outSize)
		#if final img size should be same as original this is done
		return map
        #final segmentation map

