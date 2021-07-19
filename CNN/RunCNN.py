import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Utils.Helpers import Helpers


class CNN():
	def __init(self,load_path):
		pass

	def loadCNN(self,load_path):
		# load json and create model
		json_path = load_path+'CNN.json'
		weights_path = load_path+'CNN.h5'
		json_file = open(json_path, 'r')
		loaded_model_json = json_file.read()
		json_file.close()

		#Create model from loaded json
		model = model_from_json(loaded_model_json)

		#Load model wieghts
		model.load_weights(weights_path)

		print("Loaded model from disk")
		return model

	def RunCNN(self,model,img_batch):
		category = ['Fluid', 'Defective', 'Crystal']
		prediction = model.predict(img_batch)

		# Find index of maximum value from 2D numpy array
		result = np.where(prediction == np.amax(prediction))
		# zip the 2 arrays to get the exact coordinates
		listOfCordinates = list(zip(result[0], result[1]))
		index = listOfCordinates[0][1]
		return category[index]