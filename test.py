import torch

model = torch.load('mod.pth')
print model.keys()
a = model['state_dict']


for i in range(12,14):
	for b in a.keys():
		if str(i) in b:
			c = b
			a[b.replace(str(i), str(i-5))] = a.pop(b)


#a['fc_wegiht'] = a.pop('fc_wei')
#a['fc_bias'] = a.pop('fc_bias')

print a.keys()
