import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, Lambda, LSTM
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from CNN.CNN import *

import pandas as pd
from random import seed, randint

from Utils.Helpers import Helpers


class SNN():
	def __init__(self):
		pass

	def createDS(self,path,model):
		'''
		Function that creates the csv files that 
		serve as DS for the CNN
		Arguments:
			-path: path to SNN DS dir
			-model: CNN model to predict
		'''

		img_cls = CNN()
		h = Helpers()
		steps = [1,5,10]

		imgs_path = path+'/plots'

		for volts in os.listdir(imgs_path):
			#Append Voltage dir to path
			volt_path = imgs_path+'/'+volts
			for step in os.listdir(volt_path):
				#Append Trayectory dir to path
				step_path = volt_path+'/'+step
				step_num = int(step.replace("s",""))
				for traj in os.listdir(step_path):
					#Append time steps
					traj_num = int(traj.replace("T",""))
					file_path = step_path+'/'+traj
					csv_name = path+'/'+volts+'-'+step+'-'+traj+'.csv'
					csv_file = open(csv_name, "w+")
					csv_file.write("#time,cat_index,cat,V_level\n")
					num_files_dir = len([f for f in os.listdir(file_path)
						if f.endswith('.png') 
						and os.path.isfile(os.path.join(file_path, f))])
					names = os.listdir(file_path+'/')
					names = sorted(names, key = lambda x : (len(x), x.split("-")[2]))
					for k in range(0,num_files_dir):
						volt_num = names[k].split('-')[0]
						volt_num = int(volt_num.replace("V",""))
						file_name = file_path+'/'+str(volt_num)+'V-'+str(traj_num)+'tray-'+str(k)+'step'+str(step_num)+'s.png'
						
						#Classify the image as 0, 1 or 2
						img_batch = h.preProcessImg(file_name)
						index, label = img_cls.runCNN(model,img_batch)
						csv_file.write(str(k*step_num)+','
							+str(index)+','
							+label+','
							+str(volt_num)+'\n')
					csv_file.close()

		            


	def prior(self,kernel_size, bias_size, dtype=None):
	    n = kernel_size + bias_size
	    prior_model = keras.Sequential(
	        [
	            tfp.layers.DistributionLambda(
	                lambda t: tfp.distributions.MultivariateNormalDiag(
	                    loc=tf.zeros(n), scale_diag=tf.ones(n)
	                )
	            )
	        ]
	    )
	    return prior_model


	def posterior(self,kernel_size, bias_size, dtype=None):
	    n = kernel_size + bias_size
	    posterior_model = keras.Sequential(
	        [
	            tfp.layers.VariableLayer(
	                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
	            ),
	            tfp.layers.MultivariateNormalTriL(n),
	        ]
	    )
	    return posterior_model



	def createCSNN(self,step,summary=False):
		'''
		Function that creates and compile the SNN
		args:
			-step: time memory size
		returns:
			-model: stochastic/recurrent model.
		''' 

		FEATURE_NAMES = [
			'Si',
			'V']

		inputs = {}
		for name in FEATURE_NAMES:
			inputs[name] = tf.keras.Input(shape=(1,), name=name)
		
		features = keras.layers.concatenate(list(inputs.values()))
		features = layers.BatchNormalization()(features)

		# Create hidden layers with weight uncertainty 
		#using the DenseVariational layer.
		for units in [16,32]:
			features = tfp.layers.DenseVariational(
				units=units,
				make_prior_fn=self.prior,
				make_posterior_fn=self.posterior,
				activation="sigmoid",
				)(features)
			featrues = layers.Dropout(0.2)

		# The output is deterministic: a single point estimate.
		outputs = layers.Dense(3, activation='softmax')(features)

		model = keras.Model(inputs=inputs, outputs=outputs)

		model.compile(
			loss = 'categorical_crossentropy',
			optimizer='adam'
			)
		if summary:
			model.summary()
			tf.keras.utils.plot_model(
				model = model,
				rankdir="TB",
				dpi=72,
				show_shapes=True
				)

		return model

	def createDNN(self,step,summary=False):
		'''
		Function that creates and compile the SNN
		args:
			-step: time memory size
		returns:
			-model: stochastic/recurrent model.
		''' 

		FEATURE_NAMES = [
			'Si',
			'V']

		inputs = {}
		for name in FEATURE_NAMES:
			inputs[name] = tf.keras.Input(shape=(1,), name=name)
		
		features = keras.layers.concatenate(list(inputs.values()))
		features = layers.BatchNormalization()(features)

		# Create hidden layers with weight uncertainty 
		#using the DenseVariational layer.
		for units in [16,32,64]:
			features = layers.Dense(units=units,activation="sigmoid")(features)
			featrues = layers.Dropout(0.2)

		# The output is deterministic: a single point estimate.
		outputs = layers.Dense(3, activation='softmax')(features)

		model = keras.Model(inputs=inputs, outputs=outputs)

		model.compile(
			loss = 'categorical_crossentropy',
			optimizer='adam'
			)
		if summary:
			model.summary()
			tf.keras.utils.plot_model(
				model = model,
				rankdir="TB",
				dpi=72,
				show_shapes=True
				)

		return model

	def createDSNN(self,step,summary=False):
		'''
		Function that creates and compile the SNN
		args:
			-step: time memory size
		returns:
			-model: stochastic/recurrent model.
		''' 

		FEATURE_NAMES = [
			'Si',
			'V']

		inputs = {}
		for name in FEATURE_NAMES:
			inputs[name] = tf.keras.Input(shape=(1,), name=name)
		
		features = keras.layers.concatenate(list(inputs.values()))
		features = layers.BatchNormalization()(features)

		# Create hidden layers with weight uncertainty 
		#using the DenseVariational layer.
		for units in [8,16]:
			features = layers.Dense(units=units,activation="sigmoid")(features)
			featrues = layers.Dropout(0.2)

		# The output is deterministic: a single point estimate.
		outputs = tfp.layers.DenseVariational(
				units = 3,
				make_prior_fn=self.prior,
				make_posterior_fn=self.posterior,
				activation="softmax",
				)(features)

		model = keras.Model(inputs=inputs, outputs=outputs)

		model.compile(
			loss = 'categorical_crossentropy',
			optimizer='adam'
			)
		if summary:
			model.summary()
			tf.keras.utils.plot_model(
				model = model,
				rankdir="TB",
				dpi=72,
				show_shapes=True
				)

		return model


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
				index = randint(0,len(train_labels))
				#print(index)
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

	def runSNN(self,model,inp):
		'''
		Function that runs SNN.
		Args:
			-model: SNN model object
			-inp: input state
		Returns:
			-out
		'''

		out = model.predict(inp)
		return out

	def trajectory(self,model,init,length):
		'''
		Function that runs SNN.
		Args:
			-model: model to obtain probabilities from
			-inp: dict containing input features
		Returns:
			-trajectory: list with predicted trajectories.
		'''

		trajectory = []
		for i in range(length):
			print(init)
			probs = self.runSNN(model,init)
			cat_dist = tfp.distributions.Categorical(probs=probs[0])		
			trajectory.append(int(cat_dist.sample(1)[0]))
			init['Si'] = np.array([trajectory[-1]])
			#init['V'] = np.array([randint(1,4)])
			#init['V'] = np.array([v[i]])
		return trajectory
