import torch
import sys
print(sys.argv[1])
latest = torch.load(str(sys.argv[1]))
late = latest['state_dict']



#for i in range(12,14):
	#for each in late.keys():
		#if str(i) in each:
			#del late[each]

#late['module.fc.weight'].resize_(7,53760)
#late['module.fc.weight'] = late.pop('module.fc.weight')
#late['module.fc.bias'] = late.pop('module.fc.bias')
#torch.save(latest, 'siamesesh2_latest.pth.tar')

print(latest['state_dict']['module.fc.weight'])
