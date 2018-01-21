import pandas as pd
import quat2euler


#classes for translation 3
#classes for rotation 5


confactor = 3.14/180.0
outfile = open("total0.csv", "a")
csvfile = "train1.csv"
read = pd.read_csv(csvfile)
print len(read.index)
for index in range(2, len(read.index)):
	res = read.ix[index][:]
	ref = read.ix[index-1][:]
	#df.ix[index][0] = res[0]
	#df.ix[index][1] = res[1]
	#df.ix[index][2] = res[0]
	if ref[0][:17] != res[0][:17]:
		continue
	#df.ix[index][2] = trancount
	#print counter
	#df.ix[index][3] = rotcount
	outfile.write(str(ref[0]) + ", " + str(res[0]) + ", " + str(res[1]-ref[1]) + ", " + str(res[2]-ref[2]) + ", " + str(res[3]-ref[3]) + ", " + str(res[4]-ref[4]) + ", " + str(res[5]-ref[5]) + ", " + str(res[6]-ref[6]) + "\n")
	#print outrot[0], outrot[1], outrot[2]
	#read.ix[index][5] = outrot[0]
	#read.ix[index][6] = outrot[1]
	#read.ix[index][7] = outrot[2]

	#write translation class


	#write rotation class

#read.drop('qz', axis=0)
