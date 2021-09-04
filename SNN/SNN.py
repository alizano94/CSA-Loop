import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input
import tensorflow_probability as tfp
from CNN.CNN import *

import pandas as pd

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

	def createSNN(self,step,summary=False):
		'''
		Function that creates and compile the SNN
		args:
			-step: time memory size
		returns:
			-model: stochastic/recurrent model.
		''' 

		hidden_units = [16,32]
		learning_rate = 0.001
		FEATURE_NAMES = [
			'cat_index',
			'V_level']

		inputs = {}
		for name in FEATURE_NAMES:
			inputs[name] = tf.keras.Input(shape=(1,), name=name)
		
		features = keras.layers.concatenate(list(inputs.values()))
		features = layers.BatchNormalization()(features)

		# Create hidden layers with weight uncertainty using the DenseVariational layer.
		for units in hidden_units:
			features = tfp.layers.DenseVariational(
				units=units,
				make_prior_fn=self.prior,
				make_posterior_fn=self.posterior,
				activation="sigmoid",
				)(features)

		# The output is deterministic: a single point estimate.
		outputs = layers.Dense(units=3, activation='softmax')(features)
		model = keras.Model(inputs=inputs, outputs=outputs)

		model.compile(
			loss='categorical_crossentropy',
			optimizer=tf.optimizers.Adam(),
			metrics=['accuracy']
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

	def createPBNN(self,step,summary=False):
		'''
		Function that creates and compile the SNN
		args:
			-step: time memory size
		returns:
			-model: stochastic/recurrent model.
		''' 

		def negative_loglikelihood(targets, estimated_distribution):
			return -estimated_distribution.log_prob(targets)

		hidden_units = [16,32]
		learning_rate = 0.001
		FEATURE_NAMES = [
			'cat_index',
			'V_level']

		inputs = {}
		for name in FEATURE_NAMES:
			inputs[name] = tf.keras.Input(shape=(1,), name=name)
		
		features = keras.layers.concatenate(list(inputs.values()))
		features = layers.BatchNormalization()(features)

		# Create hidden layers with weight uncertainty using the DenseVariational layer.
		for units in hidden_units:
			features = tfp.layers.DenseVariational(
				units=units,
				make_prior_fn=self.prior,
				make_posterior_fn=self.posterior,
				activation="sigmoid",
				)(features)

		# The output is deterministic: a single point estimate.
		distribution_params = layers.Dense(units=2)(features)
		outputs = tfp.layers.IndependentNormal(1)(distribution_params)
		model = keras.Model(inputs=inputs, outputs=outputs)

		model.compile(
			loss=negative_loglikelihood,
			optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
			metrics=['accuracy']
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

	def trainSNN(self,PATH,model,step,epochs=10,batch=16,plot=False):
		'''
		A function that trains a SNN given the model
		and the PATH of the data set.
		'''
		h = Helpers()

		train_features = pd.DataFrame()
		train_labels = pd.DataFrame()
		
		for file in os.listdir(PATH):
			if file.endswith('.csv'):
				train_csv = PATH+'/'+file
				X, Y = h.preProcessTens(train_csv)
				train_features = train_features.append(X)
				train_labels = train_labels.append(Y)

		print(train_features)
		print(train_labels)

		train_features_dict = {name: np.array(value)
					for name, value in train_features.items()}

		train_labels = np.array(train_labels,dtype=float)
		#train_labels_arr = np.zeros((len(train_labels),3),dtype=int)
		#for i in range(len(train_labels)):
		#	train_labels_arr[i,train_labels[i][0]] = 1 
			


		history = model.fit(train_features_dict,
			train_labels,
			epochs=epochs,
			batch_size=batch,
			validation_split=0.1,
			callbacks = [tf.keras.callbacks.EarlyStopping(
				monitor='val_loss',
				min_delta=0.01,
				patience=7
				)],
			verbose=2
			)

		if plot:
			#Plot Accuracy, change this to matplotlib
			fig = go.Figure()
			fig.add_trace(go.Scatter(x=history.epoch,
	                         y=history.history['accuracy'],
	                         mode='lines+markers',
	                         name='Training accuracy'))
			fig.add_trace(go.Scatter(x=history.epoch,
	                         y=history.history['val_accuracy'],
	                         mode='lines+markers',
	                         name='Validation accuracy'))
			fig.update_layout(title='Accuracy',
	                  xaxis=dict(title='Epoch'),
	                  yaxis=dict(title='Percentage'))
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
		#x = [inp]
		#out =[inp[0][0][0]]

		flag = False

		if flag:
			for i in range(length):
				pred = model.predict(inp)

				# Find index of maximum value from 2D numpy array
				result = np.where(pred == np.amax(pred))
				# zip the 2 arrays to get the exact coordinates
				listOfCordinates = list(zip(result[0], result[1]))
				index = listOfCordinates[0][1]
				out.append(index)

				time_stamp += t_step
				new = [[out[-1],time_stamp,vol_lvl]]
				inp = np.append(inp,np.reshape(new,(1,3,1)),axis=2)
				inp = np.delete(inp,0,axis=2)
				inp = np.reshape(inp,(1,3,step))
				x.append(inp)
		
		return out#, x
