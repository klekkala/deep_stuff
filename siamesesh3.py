import argparse
import os
import shutil
import time
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import math

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names.append('siamnet')
class SPPLayer(nn.Module):

    def __init__(self, levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.levels = levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(len(self.levels)):
            kernel_size = int(math.ceil(float(h)/float(self.levels[i])))
            kernel_stride = int(math.floor(float(h)/float(self.levels[i])))
            #print("level is ", self.levels[i])
            #print("stride is ", kernel_stride)
            #print("size is ", kernel_size)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_stride).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_stride).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()


        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  16, 2), 
            conv_dw( 16,  32, 1),
            conv_dw( 32, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            #conv_dw(256, 512, 1),
            #conv_dw(512, 512, 1),
            #nn.AvgPool2d(7),
        )

        self.spp = SPPLayer([1,2,4,10])
        self.fc = nn.Linear(53760, 7)
	#self.fc2 = nn.Linear(512, 256)
	#self.fc3 = nn.Linear(256, 7)
            #nn.BatchNorm1d(7),
    def forward_once(self, x):
        output = self.model(x)
        #print(output)
        output = self.spp(output)
        print(output)
	#output = self.fc2(output)
	#output = self.fc3(output)
        #output = nn.BatchNorm1d(7)
        #print(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        out = torch.cat((output1, output2), 1)
        total = self.fc(out)
        return total
