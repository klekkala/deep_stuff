import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from dataset import *

class PLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=10.0):
        super(PLoss, self).__init__()
        self.margin = margin

    def forward(self, total, label):
        #total = output1-output2
        #print output1
        #print output2
	#print total
	#print label
	#trans_ind = torch.cuda.LongTensor([0, 1, 2])
	#rot_ind = torch.cuda.LongTensor([3, 4, 5, 6])
	#trans_val = total.index_select(1, trans_ind)
	#rot_val = total.index_select(1, rot_ind)
        trans_val = total.narrow(1,0,3)
        rot_val = total.narrow(1,3,3)
	#print trans_val, rot_val
	#trans_label = label.index_select(1, trans_ind)
	#rot_label = label.index_select(1, rot_ind)
        trans_label = label.narrow(1,0,3)
        rot_label = label.narrow(1,3,3)
	#print trans_label
        trans_distance = F.pairwise_distance(trans_val, trans_label)
        rot_distance = F.pairwise_distance(rot_val, rot_label)
	#print "hello"
        rot = torch.mul(rot_distance.data, 4)
        #print(rot)
        #print "done", trans_distance, rot, beta
       	#print(trans_distance.data)
        loss = torch.mean((trans_distance) + Variable(rot))
        #loss = torch.mean(trans_distance + rot_distance)
	#print loss
        #loss = torch.mean(trans_distance + rot_distance)
        #print("output", output1, output2)
        #print(label)
        return loss
