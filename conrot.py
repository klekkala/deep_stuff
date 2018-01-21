import pandas as pd
import quat2euler


#classes for translation 3
#classes for rotation 5

r = [[-6, -2, 0, 2, 6], [-6, -2, 0, 2, 6], [-6, -2, 0, 2, 6]]

confactor = 3.14/180.0
outfile = open("total.csv", "w")
csvfile = "final.csv"
read = pd.read_csv(csvfile)
print len(read.index)
for index in range(1, len(read.index)):
	print index
	res = read.ix[index][:]
	#df.ix[index][0] = res[0]
	#df.ix[index][1] = res[1]
	#df.ix[index][2] = res[0]

	trancount = 0
	rotcount = 0
	for i in range(2,5):
		val = round(res[i], 2)
		if val > 0.0:
			rule = 0
		elif val < 0.0:
			rule = 2
		elif val == 0.0:
			rule = 1
		trancount += rule*pow(3, 4-i)
	#df.ix[index][2] = trancount
	inquat = [res[5], res[6], res[7], res[8]] 
	#inquat = res[5:8]
	#print inquat
	#read.ix[index][:] = read.ix[index][:7]
	outrot = quat2euler.quat2euler(inquat)
	minval = 200
	minind = -5
	counter = 0
	for i in range(0,3):
		for j in range(0, 5):
			if abs(confactor*r[i][j] - outrot[i]) < minval:
				minval = abs(confactor*r[i][j] - outrot[i])
				minind = j
				#counter += 1
		minval = 200
		rotcount += minind*pow(5, 2-i)
		minind = -5
	#print counter
	#df.ix[index][3] = rotcount
	outfile.write(str(res[0]) + ", " + str(res[1]) + ", " + str(trancount) + ", " + str(rotcount) + "\n")
	#print outrot[0], outrot[1], outrot[2]
	#read.ix[index][5] = outrot[0]
	#read.ix[index][6] = outrot[1]
	#read.ix[index][7] = outrot[2]

	#write translation class


	#write rotation class

#read.drop('qz', axis=0)
