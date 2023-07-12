import torch
import os


# determine the device to be used for training and evaluation
# determine if we will be pinning memory during data loading

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.0002
NUM_EPOCHS = 400 #just do 1 for now so we can try training on holly
BATCH_SIZE = 4
# define the input image dimensions
INPUT_IMAGE_WIDTH = 512
INPUT_IMAGE_HEIGHT = 512

IN_CHAN = 1
OUT_CHAN = 1
BASE_DIM = 32
DROP_FAC = 0
KER_SIZE = 2
NO_GROUPS = 32

ITNET_ITS = 4