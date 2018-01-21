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
from siamese import *
from loss import *
import pandas as pd


class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset, csvfile, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.csvfile = csvfile
        self.transform = transform
        
    def __getitem__(self,index):
        
        read = pd.read_csv(self.csvfile)
        res = read.ix[index][:]

        impath = os.path.join(self.imageFolderDataset)
        img0 = Image.open(impath + res[0].replace(" ", ""))
        img1 = Image.open(impath + res[1].replace(" ", ""))

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
       
        return img0, img1, torch.from_numpy(np.array([res[2], res[3], res[4], res[5], res[6], res[7]],dtype=np.float32))
    
    def __len__(self):
        read = pd.read_csv(self.csvfile)
        return len(read.index)
