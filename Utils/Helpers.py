import numpy as np
import matplotlib.pyplot as plt

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
		for i in range(len(data)-step):
			d=i+step  
			X.append(data[i:d,])
			Y.append(data[d,])
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