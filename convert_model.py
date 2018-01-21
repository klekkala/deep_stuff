import torch
from torch.autograd import Variable as V
import torchvision.models as models
from PIL import Image
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import mobilenet
# th architecture to use
arch = 'mobilenet'

# create the network architecture
#model = models.__dict__[arch](num_classes=365)
model = mobilenet.Net()
model_weight = '%s.pth.tar' % arch

checkpoint = torch.load(model_weight, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!
state_dict = {str.replace(k.encode('utf-8'),'module.',''): v for k,v in checkpoint['state_dict'].iteritems()} # the data parallel layer will add 'module' before each layer name
model.load_state_dict(state_dict)
model.eval()

model.cpu()
torch.save(model, 'whole_' + model_weight)
print 'save to ' + 'whole_' + model_weight
