#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import os
import sys
import math

import numpy as np

train_loader = torch.utils.data.DataLoader(
        dset.ImageFolder('../final', transforms.Compose([
            transforms.ToTensor(),
        ])))
b = torch.FloatTensor(12189,3,224,224)
tt = enumerate(train_loader)

for (i, da) in tt:
	da = da[0]
	da.resize_(da.shape[1], da.shape[2], da.shape[3])
	b[i,:,:,:] = da
#print b.numpy()
data = b.numpy().astype(np.float32)

means = []
stdevs = []
for i in range(3):
    pixels = data[:,i,:,:].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print("means: {}".format(means))
print("stdevs: {}".format(stdevs))
print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
