from pyquaternion import Quaternion
import os
import numpy as np
from shutil import copyfile
from decimal import Decimal

finalfile = open("final.csv", "w")

def read_traj(filename):
	file = open(filename)
	data = file.read()
	lines = data.replace(","," ").replace("\t"," ").split("\n")
	total = []
	for each in lines:
		temp = each.split()
		if len(temp) == 8 or len(temp) == 2:
			total.append(temp)

	return total

def relative(t1, r1, t2, r2):
	#trans = []
	trans = t2 - t1
	qr1 = Quaternion(array=r1)
	qr2 = Quaternion(array=r2)
	#rot1 = qr1.rotation_matrix
	#rot2 = qr2.rotation_matrix
	#relrot = rot1.transpose().dot(rot2)
	#relquat = Quaternion(matrix=relrot)
	relquat = qr2/qr1
	val = np.concatenate((trans, relquat.elements), axis=0)
	#print val
	return val

def make_data(fdir):
	truth = read_traj(fdir + '/groundtruth.txt')
	rgb = read_traj(fdir + '/rgb.txt')

	for i in range(0, len(rgb), 6):

		for j in range(0, len(truth)):
			if truth[j][0] > rgb[i][0]:
				j -= 1
				break
		next = j+1
		if i > 0:
			val = relative(np.array([float(truth[pretruth][1]), float(truth[pretruth][2]), float(truth[pretruth][3])]), np.array([float(truth[pretruth][5]), float(truth[pretruth][6]), float(truth[pretruth][7]), float(truth[pretruth][4])]), np.array([float(truth[j][1]), float(truth[j][2]), float(truth[j][3])]), np.array([float(truth[j][5]), float(truth[j][6]), float(truth[j][7]), float(truth[j][4])]))
			finalfile.write(str(rgb[prergb][1]) + ", " + str(rgb[i][1]) + ", " + str(val[0]) + ", " + str(val[1]) + ", " + str(val[2]) + ", " + str(val[3]) + ", " + str(val[4]) + ", " + str(val[5]) + ", " + str(val[6]) + "\n")
			prergb = i
			pretruth = j
			copyfile(fdir + '/' + rgb[i][1], '../fin/' + rgb[i][1])
			#print "done"

		else:
			copyfile(fdir + '/' + rgb[i][1], '../fin/' + rgb[i][1])
			pretruth = j
			prergb = 0

if __name__ == "__main__":
###Go into each directory
	direct = [dI for dI in os.listdir('.') if os.path.isdir(os.path.join('.',dI))]
	for each in direct:
		if 'groundtruth.txt' in os.listdir(each):
			make_data(each)
		elif 'groundtruth.txt' in os.listdir(each + '/' + each):
			make_data(each + '/' +each)
		else:
			print each
