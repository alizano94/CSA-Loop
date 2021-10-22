import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from random import seed, randint

class Helpers():
	'''
	Helper functions for the loop
	'''
	def __init__(self):
		pass
		
	
	def convertToMatrix(self,data,step):
		'''
		A function that takes a series of data 
		and returns the X and Y tensors that serves
		as DS for the SNN
		args:
			-data: Numpy type array containing 
				   the data
			-step: The ammount of previous steps
				   to store in memory.
		'''
		X, Y =[], []
		for i in range(len(data[0,:])-step):
			d=i+step  
			X.append(data[:,i:d])
			Y.append(data[0,d])
		return np.array(X), np.array(Y)

	def plotImages(images_arr):
		'''
		A function that plots images in the form of 
		a grid with 1 row and 3 columns where images 
		are placed in each column.

		args: 
			-images_arr: array of images to plot
		'''
		fig, axes = plt.subplots(1, 3, figsize=(20,20))
		axes = axes.flatten()
		for img, ax in zip( images_arr, axes):
			ax.imshow(img)
			ax.axis('off')
		plt.tight_layout()
		plt.show()

	def preProcessImg(self,img_path,IMG_H=212,IMG_W=212):
		'''
		A function that preprocess an image to fit 
		the CNN input.
		args:
			-img_path: path to get image
			-IMG_H: image height
			-IMG_W: image width
		Returns:
			-numpy object containing:
				(dum,img_H,img_W,Chanell)
		'''
		#Load image as GS with st size
		img = image.load_img(img_path,color_mode='grayscale',target_size=(IMG_H, IMG_W))
		#save image to array (H,W,C)
		img_array = image.img_to_array(img)

		#Create a batch of images
		img_batch = np.expand_dims(img_array, axis=0)
		return img_batch

	def preProcessTens(self,csv,print_tensors=False):
		'''
		Function that takes data form csv 
		and creates a pd Data Frame to use as 
		'''
		dataset = pd.read_csv(csv)
		train_features = dataset.copy()
		train_features.drop(train_features.tail(1).index,inplace=True)
		train_features.drop(columns=['cat'],inplace=True)
		#train_features.drop(columns=['cat','#time'],inplace=True)

		train_labels = dataset.copy()
		train_labels.drop(columns=['cat','V_level','#time'],inplace=True)
		train_labels.drop(train_labels.head(1).index,inplace=True)		

		return train_features, train_labels

	def Data2df(self,PATH,step,window):
		'''
		Takes directory with csv and loads them into a DF
		'''
		train_features = pd.DataFrame()
		train_labels = pd.DataFrame()
		
		for file in os.listdir(PATH):
			if file.endswith('.csv'):
				train_csv = PATH+'/'+file
				X, Y = self.preProcessTens(train_csv)
				tmp_X = pd.DataFrame()#columns=['V','S1','S2'])
				tmp_Y = pd.DataFrame(columns=['S0'])
				for index, rows in X.iterrows():
					new_size = len(X) - window
					if index < new_size and index > step-1:
						x_dict = {}
						for i in range(step):
							name = 'S'+str(-i-1)
							x_dict[name] = X.at[index-i,'cat_index']
						x_dict['V'] = rows['V_level']
						tmp_X = tmp_X.append(x_dict,ignore_index=True)
						tmp_Y = tmp_Y.append(
							{'S0':Y.loc[index+window,'cat_index']
							},ignore_index=True)


				train_features = train_features.append(tmp_X)
				train_labels = train_labels.append(tmp_Y)
		train_features.reset_index(inplace=True)
		train_labels.reset_index(inplace=True)


		return train_features, train_labels

	def createhist(self,train_labels):
		'''
		create histogram from labels data
		'''
		hist = [0,0,0]

		for index, rows in train_labels.iterrows():
			hist[rows['S0']] += 1

		return hist

	def DropBiasData(self,train_features,train_labels):
		'''
		Resamples the data to ensure theres no BIAS on 
		ouput state dsitribution.
		'''
		seed(1)

		hist = self.createhist(train_labels)

		min_hist = min(hist)
		arg_min = np.argmin(hist)

		while max(hist) != min_hist:
			index = randint(0,len(train_labels)-1)
			hist_index = int(train_labels['S0'][index])
			if hist[hist_index] > min_hist:
		 		train_labels.drop(index=index, inplace=True)
		 		train_features.drop(index=index, inplace=True)
			hist = self.createhist(train_labels)
			train_labels.reset_index(inplace=True)
			train_features.reset_index(inplace=True)
			train_labels.drop(columns=['index'],inplace=True)
			train_features.drop(columns=['index'],inplace=True)
		print(hist)

		train_labels.drop(columns=['level_0'],inplace=True)
		train_features.drop(columns=['level_0'],inplace=True)

		PATH = './SNN/DS/Resampled'
		train_features_csv = PATH + '/RS_train_featrues.csv'
		train_labels_csv = PATH + '/RS_train_labels.csv'
		train_features.to_csv(train_features_csv)
		train_labels.to_csv(train_labels_csv)

		return train_features, train_labels

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

	def df2dict(self,df,dtype=float):
		'''
		Takes a df and returns a dict of tensors
		'''
		out_dict = {name: np.array(value,dtype=dtype)
					for name, value in df.items()}

		return out_dict

	def onehotencoded(slef,df,dtype=float):
		'''
		Transforms array with out state into one hot encoded vector
		'''
		array = np.array(df,dtype=dtype)
		onehotencoded_array = np.zeros((len(array),3),dtype=int)
		for i in range(len(array)):
			index = int(array[i][0])
			onehotencoded_array[i][index] = 1

		return onehotencoded_array 



	def saveWeights(self,model,save_path,name):
		model.save_weights(os.path.join(save_path, name+'.h5'))

	def loadWeights(self,load_path,model):
		'''
		Functions that loads weight for the model
		args:
			-load_path: path from which to load weights
			-model: keras model
		'''
		#Load model wieghts
		model.load_weights(load_path)
		print("Loaded model from disk")

	def csv2df(self,csvname):
		'''
		Function that extracts info from csv 
		and creates array.
		'''
		results = []
		with open(csvname) as csvfile:
			reader = csv.reader(csvfile) # change contents to floats
			headers = next(reader) 
			for row in reader:
				results.append(row)

		x = []
		for i in range(0,len(results)):
			x.append(float(results[i][1]))

		df = pd.DataFrame(x)

		return df

	def plotList(self,list,x_lab,y_lab,save_path='./out.png'):
		'''
		Function that plots elements on a list
		'''
		data = np.zeros((2,len(list)))
		for i in range(len(list)):
			data[0][i] = float(i)
			data[1][i] = list[i]
		plt.figure()
		plt.figure()
		plt.yticks([0, 1, 2], ['Fluid', 'Deffective', 'Crystal']
			,rotation=45)
		plt.ylim(-.5,2.5)
		plt.xlabel(x_lab)
		#plt.ylabel(y_lab)
		plt.ylim(-.5,2.5)
		plt.scatter(data[:][0],data[:][1],s=0.5)
		plt.savefig(save_path)