import os
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
		steps = [1,5,10]

		imgs_path = path+'/plots'

		for volts in os.listdir(imgs_path):
			#Append Voltage dir to path
			volt_path = imgs_path+'/'+volts
			volt_num = int(volts.replace("V",""))
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
					for k in range(0,num_files_dir):
						file_name = file_path+'/'+str(volt_num)+'V-'+str(traj_num)+'tray-'+str(k)+'step'+str(step_num)+'s.png'
						

						#Classify the image as 0, 1 or 2
						img_batch = h.preProcessImg(file_name)
						index, label = img_cls.runCNN(model,img_batch)
						csv_file.write(str(k*step_num)+','
							+str(index)+','
							+label+','
							+str(volt_num)+'\n')
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
		model.add(SimpleRNN(units=32, input_shape=(3,step), activation="relu"))
		model.add(tfp.layers.DenseFlipout(16, activation="relu")) #Random kernel and bias layer 
		model.add(Dense(1, activation='softmax'))

		model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['accuracy'])
		if summary:
			model.summary()

		return model

	def trainSNN(self,PATH,model,step,epochs=10,batch=16):
		'''
		A function that trains a SNN given the model
		and the PATH of the data set.
		'''
		h = Helpers()

		train_X,train_Y = np.zeros([1,3,step]),np.array([])

		
		for file in os.listdir(PATH):
			if file.endswith('.csv'):
				train_csv = PATH+'/'+file
				X, Y = h.preProcessTens(train_csv,step)
				train_X = np.append(train_X,X,axis=0)
				train_Y = np.append(train_Y,Y)
		train_X = np.delete(train_X,(0),axis=0)
		print(train_X.shape)	
		
		test_csv = PATH + '/V4-10s-T3.csv' #define better testing ser=t
		test_X, test_Y = h.preProcessTens(test_csv,step)


		history = model.fit(train_X,train_Y,
			epochs=epochs,
			batch_size=batch,
			verbose=0
			)

		XtrainScore = model.evaluate(train_X, train_Y, verbose=0)
		XtestScore = model.evaluate(test_X, test_Y, verbose=0)
		#print("Training Score")
		#print(XtrainScore)
		#print("Testing Score")
		#print(XtestScore)

	def runSNN(self,model,init,vol_lvl,time_stamp,step,length=200):
		'''
		Function that runs SNN.
		Args:
			-model: SNN model object
			-inp: input state
		Returns:
			-out
		'''
		inp = [[float(init),
				time_stamp,
				vol_lvl]]
		inp = np.transpose([inp[-1]] * step)
		inp = np.reshape(inp,(1,3,step))
		out =[init]
		for i in range(length):
			pred = model.predict(inp)
			out.append(pred[0][0])
			#print(pred)
			init = out[-1]
			time_stamp += 1
			new = [[init,time_stamp,vol_lvl]]
			inp = np.append(inp,np.reshape(new,(1,3,1)),axis=2)
			inp = np.delete(inp,0,axis=2)
			inp = np.reshape(inp,(1,3,step))
				
			
		out = [round(x) for x in out]
		return out
