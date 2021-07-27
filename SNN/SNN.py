import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
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

		imgs_path = path+'/plots'
		for i in range(3,6):
			#Append Voltage dir to path
			volt_path = imgs_path+'/V'+str(i)
			for j in range(1,6):
				#Append Trayectory dir to path
				file_path = volt_path+'/T'+str(j)
				csv_name = path+'/'+str(i)+'V-'+str(j)+'tray.csv'
				csv_file = open(csv_name, "w+")
				csv_file.write("#step,cat_index,cat,V \n")
				num_files_dir = len([f for f in os.listdir(file_path)
					if f.endswith('.png') 
					and os.path.isfile(os.path.join(file_path, f))])
				for k in range(0,num_files_dir):
					file_name = file_path+'/'+str(i)+'V-'+str(j)+'tray-'+str(k)+'step.png'
					#Classify the image as 0, 1 or 2
					img_batch = h.preProcessImg(file_name)
					index, label = img_cls.runCNN(model,img_batch)
					csv_file.write(str(k)+','+str(index)
						+','+label+','+str(i)+'\n')
				csv_file.close()

		            


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

		
		test_csv = PATH+'/5V-5tray.csv'

		train_X,train_Y = np.empty([1,step]),np.array([])


		for i in range(1,5):
			train_csv = PATH+'/5V-'+str(i)+'tray.csv'

			train_df = h.csv2df(train_csv)
			train = train_df.values
			train = np.append(train,np.repeat(train[-1,],step))
			X,Y = h.convertToMatrix(train,step)
			train_X,train_Y = np.concatenate((train_X,X),axis=0),np.append(train_Y,Y)

		train_X = np.delete(train_X,(0),axis=0)

		#Read data from csv to pandas data frame
		test_df = h.csv2df(test_csv)

		#Extract data form the frame
		test = test_df.values

		# add step elements into train and test
		test = np.append(test,np.repeat(test[-1,],step))

		#Convert data to matrix
		test_X,test_Y = h.convertToMatrix(test,step)

		#Reshape to tensor of shape (len(ds),1,step)
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