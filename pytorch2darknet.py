import torch
import torchvision
from cfg import save_conv, save_conv_bn, save_fc
import mobilenet
def save_bottlenet_weights(model, fp):
    save_conv_bn(fp, model.conv1, model.bn1)
    save_conv_bn(fp, model.conv2, model.bn2)
    save_conv_bn(fp, model.conv3, model.bn3)
    if model.downsample:
        save_conv_bn(fp, model.downsample[0], model.downsample[1])

def save_resnet_weights(model, filename):
    fp = open(filename, 'wb')
    header = torch.IntTensor([0,0,0,0])
    header.numpy().tofile(fp)
    save_conv_bn(fp, model.conv1, model.bn1)
    for i in range(len(model.layer1._modules)):
        save_bottlenet_weights(model.layer1[i], fp)
    for i in range(len(model.layer2._modules)):
        save_bottlenet_weights(model.layer2[i], fp)
    for i in range(len(model.layer3._modules)):
        save_bottlenet_weights(model.layer3[i], fp)
    for i in range(len(model.layer4._modules)):
        save_bottlenet_weights(model.layer4[i], fp)
    save_fc(fp, model.fc)
    fp.close()

def save_alexnet_weights(model, filename):
    fp = open(filename, 'wb')
    header = torch.IntTensor([0,0,0,0])
    header.numpy().tofile(fp)
    for layer in model.model:
	print layer
        if type(layer) == torch.nn.Conv2d:
            print "1"
	    print(layer)
            save_conv(fp, layer)
    #for layer in model.fc:
        
    if type(model.fc) == torch.nn.Linear:
            print(layer)
            save_fc(fp, model.fc)
    


model_name = 'mobilenet'
if model_name == 'resnet50':
    resnet50 = torchvision.models.resnet50(pretrained=True)
    print('convert pytorch resnet50 to darkent, save resnet50.weights')
    save_resnet_weights(resnet50, 'resnet50.weights')

elif model_name == 'mobilenet':
    #mobilenet = torchvision.models.alexnet(pretrained=True)
    mobilenet = mobilenet.Net()
    checkpoint = torch.load('whole_mobilenet.pth.tar')
    #mobilenet.load_state_dict(checkpoint['state_dict']) 
    print('convert pytorch alexnet to darknet, save alexnet-pytorch2darknet.weights')
    save_alexnet_weights(mobilenet, 'alexnet-pytorch2darknet.weights')
