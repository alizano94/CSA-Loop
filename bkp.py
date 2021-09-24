#############################################################################################

import numpy as np
import matplotlib.pyplot as plt

#Load weigths
	snn_load_flag = True

	if snn_load_flag:
		print('Loading SNN model...')
		load_path_snn = save_model_path+snn_weights_name+'.h5'
		aux.loadWeights(load_path_snn,snn_model)

	#Get second step
	X = range(10)
	for j in range(4):
		add = 1
		vol_lvl = j+add
		for k in range(3):
			example_dict = {'cat_index': np.array([k]),
							'V_level':np.array([vol_lvl])}

			for i in range(10):
				pred_params = trayectory.runSNN(snn_model,example_dict)
				n = pred_params[:,0]; p = pred_params[:,1]
				plt.plot(X, nbinom.pmf(X, n, p), 'o', ms=8)
			plt.xlabel('State')
			#plt.title('Binomial probabilities. \nDS: VR\nPrior State: 0\nV_lvl: 4')
			plt.ylim(bottom=0)
			plt.xlim([0.0,10.0])
			fig_name = './Results/Binoms/V'+str(add)+'DS-NTS-V'+str(vol_lvl)+'-IS'+str(k)+'.png'
			#fig_name = './Results/Binoms/VRDS-NTS-V'+str(vol_lvl)+'-IS'+str(k)+'.png'
			plt.savefig(fig_name)
			plt.clf()
			#plt.show()

############################DS Histogram#################################

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

############################DS Histogram#################################


import pandas as pd

csv = './VRand.csv'

window = 10

dataset = pd.read_csv(csv)
train_features = dataset.copy()
train_features.drop(train_features.tail(1).index,inplace=True)
train_features.drop(columns=['cat'],inplace=True)
#train_features.drop(columns=['cat','#time'],inplace=True)
train_features.rename(columns={'cat_index':'Si'},inplace=True)

train_labels = dataset.copy()
train_labels.drop(columns=['cat','V_level','#time'],inplace=True)
train_labels.drop(train_labels.head(1).index,inplace=True)
train_labels.rename(columns={'cat_index':'So'},inplace=True)

dataset = pd.concat([train_features, train_labels.reset_index()],
					axis=1)
dataset.drop(columns=['index'],inplace=True)

new_train_features = pd.DataFrame(columns=['V','Si'])
new_train_labels = pd.DataFrame(columns=['So'])

for index, rows in dataset.iterrows():
	new_size = len(dataset) - window
	if index < new_size:
		new_train_features = new_train_features.append(
			{'V': rows['V_level'],
			 'Si':rows['Si']
			},ignore_index=True)
		new_train_labels = new_train_labels.append(
			{'So':train_labels.loc[index+10,'So']
			},ignore_index=True)

print(new_train_features)
print(new_train_labels)


import pandas as pd
import numpy as np
from random import seed, randint

csv = './VRand.csv'

window = 10

dataset = pd.read_csv(csv)
train_features = dataset.copy()
train_features.drop(train_features.tail(1).index,inplace=True)
train_features.drop(columns=['cat'],inplace=True)
#train_features.drop(columns=['cat','#time'],inplace=True)
train_features.rename(columns={'cat_index':'Si'},inplace=True)

train_labels = dataset.copy()
train_labels.drop(columns=['cat','V_level','#time'],inplace=True)
train_labels.drop(train_labels.head(1).index,inplace=True)
train_labels.rename(columns={'cat_index':'So'},inplace=True)

dataset = pd.concat([train_features, train_labels.reset_index()],
					axis=1)
dataset.drop(columns=['index'],inplace=True)

new_train_features = pd.DataFrame(columns=['V','Si'])
new_train_labels = pd.DataFrame(columns=['So'])

for index, rows in dataset.iterrows():
	new_size = len(dataset) - window
	if index < new_size:
		new_train_features = new_train_features.append(
			{'V': rows['V_level'],
			 'Si':rows['Si']
			},ignore_index=True)
		new_train_labels = new_train_labels.append(
			{'So':train_labels.loc[index+10,'So']
			},ignore_index=True)

print(new_train_features)
print(new_train_labels)

hist = [0,0,0]

for index, rows in new_train_labels.iterrows():
	hist[rows['So']] += 1

seed(1)
prom = sum(hist)/len(hist)

if prom != min(hist):
	print('dropping')
	index = randint(0,len(new_train_labels))
	element = new_train_labels.loc[[index]]['So']
	if elemen != np.argmin(hist):
		new_train_labels.drop([index], inplace=True)
		new_train_features.drop([index], inplace=True)
	for index, rows in new_train_labels.iterrows():
		hist[rows['So']] += 1
	prom = sum(hist)/len(hist)

print(hist)
print(prom)