import os
import pandas as pd
import numpy as np

from Utils.Helpers import Helpers

aux = Helpers()

PATH = './SNN/test-DS/FDS'

train_features = pd.DataFrame()
train_labels = pd.DataFrame()

for file in os.listdir(PATH):
	if file.endswith('.csv'):
		train_csv = PATH+'/'+file
		X, Y = aux.preProcessTens(train_csv)
		train_features = train_features.append(X)
		train_labels = train_labels.append(Y)

train_features.rename(columns={'cat_index':'Si'},inplace=True)
train_features.reset_index(inplace=True)
train_labels.rename(columns={'cat_index':'So'},inplace=True)
train_labels.reset_index(inplace=True)


print(train_features)
print(train_labels)

dataset = pd.concat([train_features, train_labels.reset_index()],
					axis=1)
dataset.drop(columns=['index','level_0'],inplace=True)

print(dataset)

for v in [1,2,3,4]:
	for si in [0,1,2]:
		example_dict = {'Si': np.array([si]),
		                'V':np.array([v])}
		c = [0,0,0]
		for index, row in dataset.iterrows():
			if row['V_level'] == v and row['Si'] == si:
				if row['So'] == 0:
					c[0] += 1
				elif row['So'] == 1:
					c[1] += 1
				else:
					c[2] += 1
		#print(example_dict)
		#print(c)
		#print(sum(c))

  