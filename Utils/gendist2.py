import os
import pandas as pd
import numpy as np
from random import seed, randint


from Utils.Helpers import Helpers

aux = Helpers()

window = 10
PATH = './SNN/test-DS/MVR'

train_features = pd.DataFrame()
train_labels = pd.DataFrame()

for file in os.listdir(PATH):
	if file.endswith('.csv'):
		train_csv = PATH+'/'+file
		X, Y = aux.preProcessTens(train_csv)
		tmp_X = pd.DataFrame(columns=['V','Si'])
		tmp_Y = pd.DataFrame(columns=['So'])
		for index, rows in X.iterrows():
			new_size = len(X) - window
			if index < new_size:
				tmp_X = tmp_X.append(
					{'V': rows['V_level'],
					'Si':rows['cat_index']
					},ignore_index=True)
				tmp_Y = tmp_Y.append(
					{'So':Y.loc[index+10,'cat_index']
					},ignore_index=True)
		train_features = train_features.append(tmp_X)
		train_labels = train_labels.append(tmp_Y)

#train_features.rename(columns={'cat_index':'Si'},inplace=True)
train_features.reset_index(inplace=True)
#train_labels.rename(columns={'cat_index':'So'},inplace=True)
train_labels.reset_index(inplace=True)


print(train_features)
print(train_labels)

dataset = pd.concat([train_features, train_labels.reset_index()],
					axis=1)
dataset.drop(columns=['index','level_0'],inplace=True)

print(dataset)

hist = [0,0,0]

for index, rows in dataset.iterrows():
	hist[rows['So']] += 1

print(hist)

seed(1)

min_hist = min(hist)
arg_min = np.argmin(hist)

while max(hist) != min_hist:
	index = randint(0,len(dataset))
	hist_index = dataset['So'][index]
	if hist[hist_index] > min_hist:
 		dataset.drop(index=index, inplace=True)
	hist = [0,0,0]
	for index, rows in dataset.iterrows():
		hist[rows['So']] += 1
	dataset.reset_index(inplace=True)
	dataset.drop(columns=['index'],inplace=True)
	print(hist)

for v in [1,2,3,4]:
	for si in [0,1,2]:
		example_dict = {'Si': np.array([si]),
		                'V':np.array([v])}
		c = [0,0,0]
		for index, row in dataset.iterrows():
			if row['V'] == v and row['Si'] == si:
				if row['So'] == 0:
					c[0] += 1
				elif row['So'] == 1:
					c[1] += 1
				else:
					c[2] += 1
		print(example_dict)
		print(c)
		print(sum(c))

  