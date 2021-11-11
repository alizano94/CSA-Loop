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

####################################################################################3

loop_flag = False

if loop_flag:
	vol_lvl = 4
	t_step = 100

	#Run the CNN to get inital step
	img_path = './InitialStates/test.png'
	img_batch = aux.preProcessImg(img_path)

	S0, S0_cat_label = img_class.runCNN(cnn_model,img_batch)

	init_feat = {'Si': np.array([S0]),
				 'V':np.array([vol_lvl])}

	trajectory = [S0]
	
	for i in range(10):
		#print(init_feat)
		pred_dist = trayectory.runSNN(snn_model,init_feat)
		out_state = np.argmax(pred_dist[0])
		init_feat['Si'] = np.array([out_state])
		init_feat['#time'] = init_feat['#time'] + np.array([t_step])
		trajectory.append(out_state)

	out_traj = {'step':np.arange(11),
				'So':np.array(trajectory)}
	out_traj = pd.DataFrame(out_traj)
	out_traj.to_csv('./out.csv', sep='\t')

####################################################################################3


	def trainSNN(self,PATH,model,step,epochs=10,batch=1,plot=True):
		'''
		A function that trains a SNN given the model
		and the PATH of the data set.
		'''
		h = Helpers()
		window = 10

		train_features = pd.DataFrame()
		train_labels = pd.DataFrame()
		
		for file in os.listdir(PATH):
			if file.endswith('.csv'):
				train_csv = PATH+'/'+file
				X, Y = h.preProcessTens(train_csv)
				tmp_X = pd.DataFrame(columns=['V','Si','t'])
				tmp_Y = pd.DataFrame(columns=['So'])
				for index, rows in X.iterrows():
					new_size = len(X) - window
					if index < new_size:
						tmp_X = tmp_X.append(
							{'V': rows['V_level'],
							'Si':rows['cat_index'],
							't':rows['#time']
							},ignore_index=True)
						tmp_Y = tmp_Y.append(
							{'So':Y.loc[index+10,'cat_index']
							},ignore_index=True)
				train_features = train_features.append(tmp_X)
				train_labels = train_labels.append(tmp_Y)
		train_features.reset_index(inplace=True)
		train_labels.reset_index(inplace=True)

		hist = [0,0,0]

		for index, rows in train_features.iterrows():
			hist[rows['Si']] += 1

		print(hist)

		seed(1)

		drop_Si=False
		drop_So=True


		if drop_Si:
			min_hist = min(hist)
			arg_min = np.argmin(hist)

			while max(hist) != min_hist:
				index = randint(0,len(train_labels)-1)
				hist_index = train_features['Si'][index]
				if hist[hist_index] > min_hist:
			 		train_labels.drop(index=index, inplace=True)
			 		train_features.drop(index=index, inplace=True)
				hist = [0,0,0]
				for index, rows in train_features.iterrows():
					hist[rows['Si']] += 1
				train_labels.reset_index(inplace=True)
				train_features.reset_index(inplace=True)
				train_labels.drop(columns=['index'],inplace=True)
				train_features.drop(columns=['index'],inplace=True)
			print(hist)

		if drop_So:
			min_hist = min(hist)
			arg_min = np.argmin(hist)

			while max(hist) != min_hist:
				index = randint(0,len(train_labels)-1)
				#print(index)
				#print(train_features)
				#print(train_labels)
				hist_index = train_labels['So'][index]
				if hist[hist_index] > min_hist:
			 		train_labels.drop(index=index, inplace=True)
			 		train_features.drop(index=index, inplace=True)
				hist = [0,0,0]
				for index, rows in train_labels.iterrows():
					hist[rows['So']] += 1
				train_labels.reset_index(inplace=True)
				train_features.reset_index(inplace=True)
				train_labels.drop(columns=['index'],inplace=True)
				train_features.drop(columns=['index'],inplace=True)
			print(hist)

		dataset = pd.concat([train_features, train_labels.reset_index()],
					axis=1)
		dataset.drop(columns=['index','level_0'],inplace=True)

		bars = ['Fluid','Defective','Crystal']
		x_pos = np.arange(len(bars))
		plt.yticks(color='black')
		fig_path = './Results/DS-Hist/probabilities/100s/MVR/Drop'
		print('Calculating DS transition probabilities...........')
		for v in [1,2,3,4]:
			for si in [0,1,2]:
				example_dict = {'Si': np.array([si]),
				                'V':np.array([v])}
				c = [0,0,0]
				plt.xticks(x_pos, bars, color='black')
				fig_name = fig_path+'MVRDS-L-S'+str(si)+'-V'+str(v)+'.png'
				for index, row in dataset.iterrows():
					if row['V'] == v and row['Si'] == si:
						c[row['So']] += 1
				plt.bar(x_pos,c,color='red')
				plt.savefig(fig_name)
				plt.clf()
				print(example_dict)
				print(c)
				print(sum(c))	
			

		#train_labels.drop(columns=['index'],inplace=True)
		#train_features.drop(columns=['index'],inplace=True)

		#print(train_features)
		#print(train_labels)
		train_labels.drop(columns=['level_0'],inplace=True)
		train_features.drop(columns=['level_0'],inplace=True)

		train_features_dict = {name: np.array(value,dtype=float)
					for name, value in train_features.items()}

		train_labels = np.array(train_labels,dtype=float)
		train_labels_arr = np.zeros((len(train_labels),3),dtype=int)
		#print(train_labels)
		for i in range(len(train_labels)):
			index = int(train_labels[i][0])
			train_labels_arr[i][index] = 1 
			


		history = model.fit(train_features_dict,
			train_labels_arr,
			epochs=epochs,
			batch_size=batch,
			validation_split=0.2,
			verbose=2
			)

		if plot:
			#Plot Accuracy, change this to matplotlib
			fig = go.Figure()
			fig.add_trace(go.Scatter(x=history.epoch,
	                         y=history.history['loss'],
	                         mode='lines+markers',
	                         name='Training accuracy'))
			fig.add_trace(go.Scatter(x=history.epoch,
	                         y=history.history['val_loss'],
	                         mode='lines+markers',
	                         name='Validation accuracy'))
			fig.update_layout(title='Loss',
	                  xaxis=dict(title='Epoch'),
	                  yaxis=dict(title='Loss'))
			fig.show()


	def trainSNNsingleFeat(self,PATH,model,step,epochs=10,batch=16,plot=False):
		'''
		A function that trains a SNN given the model
		and the PATH of the data set.
		'''
		h = Helpers()
		window = 10

		train_features = pd.DataFrame()
		train_labels = pd.DataFrame()
		
		for file in os.listdir(PATH):
			if file.endswith('.csv'):
				train_csv = PATH+'/'+file
				X, Y = h.preProcessTens(train_csv)
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
		train_features.reset_index(inplace=True)
		train_labels.reset_index(inplace=True)

		hist = [0,0,0]

		index_list = []

		for index, rows in train_features.iterrows():
			if rows['Si'] != 1 or rows['V'] != 4:
				index_list.append(index)
			print(index_list)
		train_labels.drop(index=index_list, inplace=True)
		train_features.drop(index=index_list, inplace=True)
		train_labels.reset_index(inplace=True)
		train_features.reset_index(inplace=True)
		train_labels.drop(columns=['index'],inplace=True)
		train_features.drop(columns=['index'],inplace=True)

		for index, rows in train_labels.iterrows():
					hist[rows['So']] += 1

		dataset = pd.concat([train_features, train_labels.reset_index()],
					axis=1)
		dataset.drop(columns=['index','level_0'],inplace=True)

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

		#train_labels.drop(columns=['index'],inplace=True)
		#train_features.drop(columns=['index'],inplace=True)

		#print(train_features)
		#print(train_labels)
		train_labels.drop(columns=['level_0'],inplace=True)
		train_features.drop(columns=['level_0'],inplace=True)

		train_features_dict = {name: np.array(value,dtype=float)
					for name, value in train_features.items()}

		train_labels = np.array(train_labels,dtype=float)
		train_labels_arr = np.zeros((len(train_labels),3),dtype=int)
		#print(train_labels)
		for i in range(len(train_labels)):
			index = int(train_labels[i][0])
			train_labels_arr[i][index] = 1 
			


		history = model.fit(train_features_dict,
			train_labels_arr,
			epochs=epochs,
			batch_size=batch,
			validation_split=0.1,
			verbose=2
			)

		if plot:
			#Plot Accuracy, change this to matplotlib
			fig = go.Figure()
			fig.add_trace(go.Scatter(x=history.epoch,
	                         y=history.history['loss'],
	                         mode='lines+markers',
	                         name='Training accuracy'))
			fig.add_trace(go.Scatter(x=history.epoch,
	                         y=history.history['val_loss'],
	                         mode='lines+markers',
	                         name='Validation accuracy'))
			fig.update_layout(title='Loss',
	                  xaxis=dict(title='Epoch'),
	                  yaxis=dict(title='Loss'))
			fig.show()

##Obtain probability tensor###
prob_tens = np.empty([4,3,3])
for v in [1,2,3,4]:
	for S0 in [0,1,2]:
		init_feat = {'V':np.array([v])}
		for i in range(step):
			name = 'S'+str(i-step)
			init_feat[name] = np.array([S0])
		#print(init_feat)
		probs = trayectory.runSNN(snn_model,init_feat)
		print(sum(probs[0]))
		prob_tens[v-1,S0,:] = probs[0]
print(prob_tens)


###########################Miscelanious####################################
example_dict = {'S-4':np.array([0]),
				'S-3':np.array([1]),
				'S-2':np.array([1]),
				'S-1':np.array([1]),
				'V':np.array([4])}
print(example_dict)

fig_name = './test.png'
n = 1000
probs = trayectory.runSNN(snn_model,example_dict)
cat_dist = tfp.distributions.Categorical(probs=probs[0])
empirical_prob = tf.cast(
	tf.histogram_fixed_width(
		cat_dist.sample(int(n)),
		[0, 2],
		nbins=3
		),dtype=tf.float32) / n

print('Model Learned Probs :\n',probs)
print('Sampled probs :\n',empirical_prob)

bars = ['Fluid','Defective','Crystal']
x_pos = np.arange(len(bars))
plt.xticks(x_pos, bars, color='black')
plt.yticks(color='black')
plt.bar(x_pos,empirical_prob,color='black')
#plt.bar(x_pos,probs[0],color='black')
plt.savefig(fig_name)

def DataTrasnProbPlot(self,train_features,train_labels,fig_path):
		'''
		plot the transition probabilities fom the dataset
		'''
		dataset = pd.concat([train_features, train_labels.reset_index()],
					axis=1)
		dataset.drop(columns=['index'],inplace=True)

		bars = ['Fluid','Defective','Crystal']
		x_pos = np.arange(len(bars))
		plt.yticks(color='black')
		print('Calculating DS transition probabilities...........')
		for v in [1,2,3,4]:
			for si in [0,1,2]:
				example_dict = {'Si': np.array([si]),
				                'V':np.array([v])}
				c = [0,0,0]
				plt.xticks(x_pos, bars, color='black')
				fig_name = fig_path+'MVRDS-L-S'+str(si)+'-V'+str(v)+'.png'
				for index, row in dataset.iterrows():
					if row['V'] == v and row['S-1'] == si:
						c[row['S0']] += 1
				plt.bar(x_pos,c,color='red')
				plt.savefig(fig_name)
				plt.clf()
				print(example_dict)
				print(c)
				print(sum(c))


