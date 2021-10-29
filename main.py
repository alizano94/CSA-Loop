import os

import tensorflow as tf
import tensorflow_probability as tfp


from Utils.Helpers import *
from CNN.CNN import *
from SNN.SNN import *

#load classes
Helpers = Helpers()
CNN = CNN()
SNN = SNN()


#Define Variables
#FLAGS
cnn_train = False
snn_train = False
preprocess_snnDS = False
Test_CNN = True
Test_SNN = False


#Parameters
memory = 1
window = 100

#paths
cnn_ds_dir = './CNN/DS/'
snn_ds_dir = '/home/lizano/Documents/CSA-Loop/SNN/DS/'
csv_snnDS_path = 'Balanced-W'+str(window)+'-M'+str(memory)+'.csv'
csv_snnDS_path = snn_ds_dir+csv_snnDS_path
weights_dir = './SavedModels/'
cnn_weights = weights_dir+'CNN.h5'
snn_weights = weights_dir+'SNN.h5'

#CNN
#Create CNN model
cnn_model = CNN.createCNN(summary=True)
#Tain the model or load learning
if cnn_train:
	if os.path.isfile(cnn_weights):
		os.remove(cnn_weights)
	CNN.trainCNN(cnn_ds_dir,cnn_model,epochs=100)
	#Check arguments for this method
	Helpers.saveWeights(cnn_model,weights_dir,'CNN')
else:
	print('Loading CNN model...')
	Helpers.loadWeights(cnn_weights,cnn_model)


#SNN
#Create SNN model
snn_model = SNN.createRNN(memory,summary=True)
#Data Set Hadelling
if os.path.isfile(csv_snnDS_path):
	pass
else:
	if preprocess_snnDS:
		Helpers.preProcessSNNDS(snn_ds_dir,cnn_model)
	SNN.createDS(snn_ds_dir,window,memory)
#Tain the model or load learning
if snn_train:
	if os.path.isfile(snn_weights):
		os.remove(snn_weights)
	SNN.trainModel(csv_snnDS_path,snn_model,epochs=100,batch=4)
	#Check arguments for this method
	Helpers.saveWeights(snn_model,weights_dir,'SNN')
else:
	print('Loading SNN model...')
	Helpers.loadWeights(snn_weights,snn_model)


#Test Data 
#Test CNN accuracy.
if Test_CNN:
	CNN.testCNN(snn_ds_dir)


