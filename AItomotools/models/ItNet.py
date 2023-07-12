from AItomotools.utils.math import power_method
from AItomotools.utils.parameter import Parameter
import AItomotools.CTtools.ct_geometry as ct

import numpy as np

import tomosipo as ts
from tomosipo.torch_support import to_autograd
from ts_algorithms import  fbp

import torch
import torch.nn as nn
import torch.nn.functional as F


class IterativeNet(InvNet): #the ItNet algorithm
    def __init__(
        self,
        subnet,
        num_iter,
        lam,
        lam_learnable=True,
        final_dc=True,
        resnet_factor=1.0,
        inverter=None,
        dc_operator=None,
        use_memory=False,
    ): #NN parameters
        
        super(IterativeNet, self).__init__()
        if isinstance(subnet, list):
            #is subnet an instance of list class?
            self.subnet = torch.nn.ModuleList(subnet)
            #gets sub network from pyTorch - how does it know which?
        else:
            self.subnet = subnet 
        self.num_iter = num_iter 
        self.final_dc = final_dc
        self.resnet_factor = resnet_factor
        self.inverter = inverter
        self.dc_operator = dc_operator
        self.use_memory = use_memory
        if not isinstance(lam, (list, tuple)):
            lam = [lam] * num_iter
        if not isinstance(lam_learnable, (list, tuple)):
            lam_learnable = [lam_learnable] * len(lam)
            #??

        self.lam = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.tensor(lam[it]), requires_grad=lam_learnable[it]
                )
                for it in range(len(lam))
            ]
        )
        #??

    def forward(self, inp):
        #forward algorithm

        x, y = inp  # get sinogram and fbp - wrong way round
        
        #can take fbp from ts_algorithms
        #sinogram is g


        #x and xinv are images
        #y is a sinogram
        #forward algo takes img and goes to data
        #backwards algo goes from data to img
        #so inverter takes sinogram (data) and goes to img 
        #so inverter is back proj

        if self.inverter is not None:
            xinv = self.inverter(y)
        else:
            xinv = x

        if self.use_memory is not False:
            #use memory passes previous iterations of img to next layer
            x_shape = xinv.shape
            s = torch.zeros(
                x_shape[0],
                self.use_memory,
                x_shape[2],
                x_shape[3],
                device=xinv.device,
            )

        for it in range(self.num_iter):

            if self.use_memory is not False:
                if isinstance(self.subnet, torch.nn.ModuleList):
                    out = self.subnet[it](torch.cat([xinv, s], dim=1))
                else:
                    out = self.subnet(torch.cat([xinv, s], dim=1))
                    #using previous outputs concatenated with xinv
                xinv = self.resnet_factor * xinv + out[:, 0:1, ...]
                #xinv is next iteration reconstruction 
                s = out[:, 1:, ...] #s is some combo of all previous iterations
            else:
                if isinstance(self.subnet, torch.nn.ModuleList):
                    #sees how mnay iterations in unet and updates xinv
                    #line 763 to find subnet
                    #look at unet networks
                    xinv = self.resnet_factor * xinv + self.subnet[it](xinv)
                else:
                    xinv = self.resnet_factor * xinv + self.subnet(xinv)

            if (self.final_dc) or (
                (not self.final_dc) and it < self.num_iter - 1
            ):
                if self.dc_operator is not None:
                    xinv = xinv - self.lam[it] * self.dc_operator((y, xinv))
                    #step that finds new img
                    #Xnew = Xold + FBP(Fx - y)
                    #This is A^T
                    #Fx is forward proj of x (an img), y is sinogram
                    #if it was grad descent not this, would use F^T and minimise diff

                    #dc operator is just fbp taken from tomosipo
                    #look at comment on LPD algo re. operators

                    #go through ad replace dc operator and self.inverse 

                    #subnet is unet
                    #unet is trained first, then this is used to build itnet

        return xinv
        #returns inverse from forward algo - i.e. gives estimated data params


    def set_learnable_iteration(self, index):
        for i in list(range(self.get_num_iter_max())):
            if i in index:
                self.lam[i].requires_grad = True
                self.subnet[i].unfreeze()
            else:
                self.lam[i].requires_grad = False
                self.subnet[i].freeze()

    def get_num_iter_max(self):
        return len(self.lam)

    def _print_info(self):
        print("Current lambda(s):")
        print(
            [
                self.lam[it].item()
                for it in range(len(self.lam))
                if self.lam[it].numel() == 1
            ]
        )
        print([self.lam[it].requires_grad for it in range(len(self.lam))])
        print("Epoch done", flush=True)
#but where does it actually run anything?


#Qs for Ander:
#- where does ItNet code actually *do* anything?
#- how to read code written like this?
#- class in a class?
#- static method?
#- how to go from paper to code?