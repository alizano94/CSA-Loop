import os
import csv
import pandas as pd

from Helpers import Helpers

h = Helpers()

c = [[-1,-1,-1,-1,-1,-1,-1,-1,-1,
	 -1,-1,-1,-1,-1,-1,-1,-1,-1,
	 -1,-1,-1,-1,-1,-1,-1,-1,-1,
	 -1,-1,-1,-1,-1,-1,-1,-1,-1,],
	 [-1,-1,-1,-1,-1,-1,-1,-1,-1,
	 -1,-1,-1,-1,-1,-1,-1,-1,-1,
	 -1,-1,-1,-1,-1,-1,-1,-1,-1,
	 -1,-1,-1,-1,-1,-1,-1,-1,-1,],
	 [-1,-1,-1,-1,-1,-1,-1,-1,-1,
	 -1,-1,-1,-1,-1,-1,-1,-1,-1,
	 -1,-1,-1,-1,-1,-1,-1,-1,-1,
	 -1,-1,-1,-1,-1,-1,-1,-1,-1,],
	 [0,0,0,0,0,0,0,0,0,
	 0,0,0,0,0,0,0,0,0,
	 0,0,0,0,0,0,0,0,0,
	 0,0,0,0,0,0,0,0,0]]

data_hist = [0,0,0]

train_features = pd.DataFrame()
train_labels = pd.DataFrame()

PATH = '../SNN/DS'


for file in os.listdir(PATH):
	if file.endswith('.csv'):
		csv = PATH+'/'+file
		train_csv = PATH+'/'+file
		X, Y = h.preProcessTens(train_csv)
		train_features = train_features.append(X)
		train_labels = train_labels.append(Y)

train_features.rename(columns={'cat_index':'Si'},inplace=True)
train_labels.rename(columns={'cat_index':'So'},inplace=True)

print(train_features)
print(train_labels)
print('DEBUG')

dataset = pd.concat([train_features.reset_index(), train_labels.reset_index()],
					axis=1)
dataset.drop(columns=['index'],inplace=True)
print(dataset)

print(dataset['Si'].dtypes)

for index, row in dataset.iterrows():
	for si in [0,1,2]:
		for v in [0,1,2,3]:
			for so in [0,1,2]:
				c[0][so+3*v+12*si] = si
				c[1][so+3*v+12*si] = v+1
				c[2][so+3*v+12*si] = so
				if row['Si']==si:
					if row['V_level'] == v+1:
						if row['So'] == so:
							c[3][so+3*v+12*si] += 1
							data_hist[row['So']] +=1
print(c)
print(sum(c[3][:]))
print(data_hist)
print(sum(data_hist))