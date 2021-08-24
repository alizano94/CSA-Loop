import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing import image

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

	def preProcessTens(self,csv,step,print_tensors=False):
		'''
		Function that takes data form csv 
		and creates a tensor to use as 
		'''
		tensor_X,tensor_Y = np.zeros([1,3,step]),np.array([])
		df = pd.read_csv(csv)
		array = df[['cat_index','#time','V_level']].to_numpy()
		array = array.T
		array = np.append(array,np.transpose([array[:,-1]] * step),axis=1)
		X,Y = self.convertToMatrix(array,step)
		tensor_X,tensor_Y = np.concatenate((tensor_X,X),axis=0),np.append(tensor_Y,Y)
		tensor_X = np.delete(tensor_X,(0),axis=0)
		np.reshape(tensor_X, (tensor_X.shape[0], 3, tensor_X.shape[2]))
		if print_tensors:
			print(tensor_X)
			print(tensor_Y)

		return tensor_X, tensor_Y

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