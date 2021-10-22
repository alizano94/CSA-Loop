import csv
import pandas as pd
import numpy as np
from Utils.Helpers import Helpers



step = 2
h = Helpers()

train_X,train_Y = np.empty([1,step]),np.array([])
print(train_X)

for i in range(1,3):
	csvname = './test'+str(i)+'.csv'
	results = []
	with open(csvname) as csvfile:
		reader = csv.reader(csvfile) # change contents to floats
		headers = next(reader) 
		for row in reader:
			results.append(row)

	x = []
	for i in range(0,len(results)):
		x.append(float(results[i][1]))

	df = pd.DataFrame(x)



	#Extract data form the frame to column vector
	train = df.values



	# add step elements into train and test
	#and reshape to row vector
	train = np.append(train,np.repeat(train[-1,],step))

	#Convert data to matrix
	#train_X matrix of shape (len(train),step)
	#train_Y row vector 
	X,Y = h.convertToMatrix(train,step)
	#print(X)
	train_X,train_Y = np.concatenate((train_X,X),axis=0),np.append(train_Y,Y)

print(train_X)
train_X = np.delete(train_X,(0),axis=0)
print(train_X)

#Reshape to tensor for model input
#Reshape to tensor of shape (len(ds),1,step)
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
print(train_X)
print(train_Y)
