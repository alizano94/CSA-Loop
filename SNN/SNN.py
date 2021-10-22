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

from Utils.Helpers import Helpers
from sklearn.preprocessing import MinMaxScaler


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
			'V',
			't']

		inputs = {}
		for name in FEATURE_NAMES:
			inputs[name] = tf.keras.Input(shape=(1,), name=name)
		
		features = keras.layers.concatenate(list(inputs.values()))
		features = layers.BatchNormalization()(features)

		# Create hidden layers with weight uncertainty 
		#using the DenseVariational layer.
		for units in [16,32,64]:
			features = layers.Dense(units=units,activation="sigmoid")(features)
			features = layers.Dropout(0.2)(features)

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

	def createRNN(self,step,summary=False,keep_v=False):
		'''
		Function that creates and compile the SNN
		args:
			-step: time memory size
		returns:
			-model: stochastic/recurrent model.
		''' 

		FEATURE_NAMES = []

		for i in range(step):
			name = 'S'+str(i-step)
			FEATURE_NAMES += [name]

		if keep_v:
			for i in range(step):
				name = 'V'+str(i-step)
		else:
			FEATURE_NAMES += ['V']


		inputs = {}
		for name in FEATURE_NAMES:
			inputs[name] = tf.keras.Input(shape=(1,), name=name)
		print(inputs)

		if keep_v:
			memory = inputs.copy()
			for i in range(step):
				name = 'V'+str(i-step)
				memory.pop(name)
		else:
			memory = inputs.copy()
			memory.pop('V')

		features = keras.layers.concatenate(list(memory.values()))
		features = layers.BatchNormalization()(features)
		features = tf.keras.layers.Reshape((step,1), input_shape=(2,))(features)
		for units in [16,32]:
			features = LSTM(units,
				activation="tanh",
				recurrent_activation="sigmoid",
				dropout=0.2,
				recurrent_dropout=0.2)(features)
			features = tf.keras.layers.Reshape((units,1))(features)

		features = keras.layers.Flatten()(features)
		features = keras.layers.concatenate([features,inputs['V']])
		features = layers.BatchNormalization()(features)

		# Create hidden layers with weight uncertainty 
		#using the DenseVariational layer.
		for units in [64,128]:
			features = layers.Dense(units=units,activation="sigmoid")(features)
			features = layers.Dropout(0.2)(features)

		# The output is deterministic: a single point estimate.
		outputs = layers.Dense(3, activation='softmax')(features)

		model = keras.Model(inputs=inputs, outputs=outputs)

		#opt = keras.optimizers.Adam(learning_rate=0.01)


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

		featrues_Si = layers.BatchNormalization()(inputs['Si'])
		featrues_V = layers.BatchNormalization()(inputs['V'])
		
		for units in [8,16]:
			featrues_Si = tfp.layers.DenseVariational(
				units = units,
				make_prior_fn=self.prior,
				make_posterior_fn=self.posterior,
				activation="softmax",
				)(featrues_Si)
			featrues_Si = layers.Dropout(0.2)(featrues_Si)

		for units in [8,16]:
			featrues_V = layers.Dense(units=units,activation="sigmoid")(featrues_V)
			featrues_V = layers.Dropout(0.2)(featrues_V)

		features = keras.layers.concatenate([featrues_Si,featrues_V])

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

	def trainModel(self,PATH,model,step,epochs=10,batch=5,plot=True):
		'''
		A function that trains a SNN given the model
		and the PATH of the data set.
		'''
		h = Helpers()
		window = 10

		load_resample = False
		if load_resample:
			PATH = './SNN/DS/Resampled'
			train_features_csv = PATH + '/RS_train_featrues.csv'
			train_labels_csv = PATH + '/RS_train_labels.csv'
			train_features = pd.read_csv(train_features_csv,index_col=0)
			train_labels =  pd.read_csv(train_labels_csv,index_col=0)
		else:
			train_features, train_labels = h.Data2df(PATH,step,window)
		hist = h.createhist(train_labels)
		print(hist)

		if load_resample==False:
			train_features, train_labels = h.DropBiasData(train_features,train_labels)

		plot_trans = False
		if plot_trans:
			fig_path = './Results/DS-Hist/probabilities/100s/MVR/Drop'
			h.DataTrasnProbPlot(train_features,train_labels,fig_path)

		train_features = h.df2dict(train_features)
		train_labels = h.onehotencoded(train_labels)

		print(train_features)
		print(train_labels)

		history = model.fit(train_features,
			train_labels,
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

	def trajectory(self,step,model,init,length):
		'''
		Function that runs SNN.
		Args:
			-model: model to obtain probabilities from
			-inp: dict containing input features
		Returns:
			-trajectory: list with predicted trajectories.
		'''

		trajectory = []
		v_traj = []
		for i in range(length):
			print(init)
			v_traj.append(init['V'])
			probs = self.runSNN(model,init)
			cat_dist = tfp.distributions.Categorical(probs=probs[0])		
			So = cat_dist.sample(1)[0]
			
			trajectory.append(int(So))
			init['S-1'] = np.array([So])

			for i in range(step-1):
				name = 'S'+str(i-step)
				past_state = 'S'+str(i-step+1)
				init[name] = init[past_state]
		return trajectory, v_traj
