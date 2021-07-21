import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import tensorflow_probability as tfp

import pandas as pd

from Utils.Helpers import Helpers


class SNN():
	def __init__(self):
		pass

	def createSNN(self,step,summary=False):
		'''
		Function that creates and compile the SNN
		args:
			-step: time memory size
		returns:
			-model: stochastic/recurrent model.
		'''
		#Create the model
		#SimpleRNN model 
		model = Sequential()
		model.add(SimpleRNN(units=32, input_shape=(1,step), activation="relu"))
		model.add(tfp.layers.DenseFlipout(16, activation="relu")) #Random kernel and bias layer 
		model.add(Dense(1))

		model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['accuracy'])
		if summary:
			model.summary()

		return model

	def trainSNN(self,PATH,model,step,epochs=3,batch=16):
		'''
		A function that trains a SNN given the model
		and the PATH of the data set.
		'''
		h = Helpers()

		train_csv = PATH+'/5V-1tray.csv'
		test_csv = PATH+'/5V-2tray.csv'

		#Read data from csv to pandas data frame
		train_df = h.csv2df(train_csv)
		test_df = h.csv2df(test_csv)

		#Extract data form the frame
		train = train_df.values
		test = test_df.values

		# add step elements into train and test
		train = np.append(train,np.repeat(train[-1,],step))
		test = np.append(test,np.repeat(test[-1,],step))

		#Convert data to matrix
		train_X,train_Y = h.convertToMatrix(train,step)
		test_X,test_Y = h.convertToMatrix(test,step)

		#Reshape to tensor for model input
		train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
		test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))


		history = model.fit(train_X,train_Y,
			epochs=epochs,
			batch_size=batch,
			verbose=2
			)

		XtrainScore = model.evaluate(train_X, train_Y, verbose=0)
		XtestScore = model.evaluate(test_X, test_Y, verbose=0)
		print("Training Score")
		print(XtrainScore)
		print("Testing Score")
		print(XtestScore)

	def runSNN(self,model,inp,length=1000):
		'''
		Function that runs SNN.
		Args:
			-model: SNN model object
			-inp: input state
		Returns:
			-out
		'''
		inp = float(inp)
		out =[inp]
		for i in range(length):
			inp = np.reshape(inp,(1,1,1))
			out.append(model.predict(inp)[0][0])
			inp = out[-1]
		out = [round(x) for x in out]
		return out