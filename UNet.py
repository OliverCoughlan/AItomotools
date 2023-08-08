from AItomotools.utils.math import power_method
from AItomotools.utils.parameter import Parameter
import AItomotools.CTtools.ct_geometry as ct

import numpy as np

import tomosipo as ts
from tomosipo.torch_support import to_autograd
from ts_algorithms import fbp #importing conventional fdk which is used in unet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop

from collections import OrderedDict
import config


class UNet(nn.Module):
    def __init__(self, inChan=config.IN_CHAN, outChan=config.OUT_CHAN, 
    baseDim=config.BASE_DIM, dropFac=config.DROP_FAC, 
    kerSiz = config.KER_SIZE, noGrp=config.NO_GROUPS):
        super(UNet, self).__init__()

        self.encoder1 = self.block(inChan,baseDim,noGrp,dropFac, "enc1")
        self.pool1 = nn.MaxPool2d(kerSiz, stride=2)
        self.encoder2 = self.block(baseDim,baseDim * 2,noGrp,dropFac,"enc2")
        self.pool2 = nn.MaxPool2d(kerSiz, stride=2)
        self.encoder3 = self.block(baseDim * 2,baseDim * 4,noGrp,dropFac,"enc3")
        self.pool3 = nn.MaxPool2d(kerSiz, stride=2)
        self.encoder4 = self.block(baseDim * 4,baseDim * 8,noGrp,dropFac,"enc4")
        self.pool4 = nn.MaxPool2d(kerSiz, stride=2)

        self.base = self.block(baseDim * 8,baseDim * 16,noGrp,dropFac,name="base")

        self.upconv4 = nn.ConvTranspose2d(baseDim * 16,baseDim * 8,kerSiz,stride=2)
        self.decoder4 = self.block(baseDim * 16,baseDim * 8,noGrp,dropFac,name="dec4")
        self.upconv3 = nn.ConvTranspose2d(baseDim * 8,baseDim * 4,kerSiz,stride=2)
        self.decoder3 = self.block(baseDim * 8,baseDim * 4,noGrp,dropFac,name="dec3")
        self.upconv2 = nn.ConvTranspose2d(baseDim * 4,baseDim * 2,kerSiz,stride=2)
        self.decoder2 = self.block(baseDim * 4,baseDim * 2,noGrp,dropFac,name="dec2")
        self.upconv1 = nn.ConvTranspose2d(baseDim * 2,baseDim, kerSiz,stride=2)
        self.decoder1 = self.block(baseDim * 2,baseDim,noGrp,dropFac,name="dec1")

        self.conv = nn.Conv2d(baseDim,outChan,kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        base = self.base(self.pool4(enc4))

        dec4 = self.upconv4(base)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    def block(self, inChan, outChan, noGrp, dropFac, name):
        return nn.Sequential(
            OrderedDict(
                [(name + "conv1",nn.Conv2d(inChan,outChan,kernel_size=3,padding=1,bias=True)),
                 (name + "gn1", nn.GroupNorm(noGrp, outChan)),
                 (name + "relu1", nn.ReLU(True)),
                 (name + "dr1", nn.Dropout(p=dropFac)),
                 (name + "conv2",nn.Conv2d(outChan,outChan,kernel_size=3,padding=1,bias=True)),
                 (name + "gn1",nn.GroupNorm(noGrp, outChan)),
                 (name + "relu2", nn.ReLU(True)),
                 (name + "dr2", nn.Dropout(p=dropFac))])
            )
