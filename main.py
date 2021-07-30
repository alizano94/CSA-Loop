import numpy as np


from Utils.Helpers import Helpers
from CNN.CNN import *
from SNN.SNN import *

#Load necesary modules
aux = Helpers()
img_class = CNN()
trayectory = SNN()

#Define variables
cnn_ds_path = './CNN/DS'
snn_ds_path = './SNN/DS'
save_model_path = './SavedModels/'
cnn_weights_name = 'CNN'
snn_weights_name = 'SNN'
step = 1

#Create the models
cnn_model = img_class.createCNN()
snn_model = trayectory.createSNN(step)


#Train the models
cnn_training_flag = False
snn_training_flag = True

if cnn_training_flag:
	img_class.trainCNN(cnn_ds_path,cnn_model)
	aux.saveWeights(cnn_model,save_model_path,cnn_weights_name)

if snn_training_flag:
	trayectory.trainSNN(snn_ds_path,snn_model,step)
	aux.saveWeights(snn_model,save_model_path,snn_weights_name)

#Load weigths
cnn_load_flag = True
snn_load_flag = True

if cnn_load_flag:
	print('Loading CNN model...')
	load_path_cnn = save_model_path+cnn_weights_name+'.h5'
	aux.loadWeights(load_path_cnn,cnn_model)
if snn_load_flag:
	print('Loading SNN model...')
	load_path_snn = save_model_path+snn_weights_name+'.h5'
	aux.loadWeights(load_path_snn,snn_model)

#Create SNN DS
snn_ds_create_flag = False
if snn_ds_create_flag:
	trayectory.createDS(snn_ds_path,cnn_model)

#Test NN
cnn_test_flag = False

if cnn_test_flag:
	print(img_class.testCNN(cnn_model,cnn_ds_path))

#Run the loop
loop_flag = True

if loop_flag:
	#Run the CNN to get inital step
	img_path = './test.png'
	img_batch = aux.preProcessImg(img_path)

	initial_step, initial_step_label = img_class.runCNN(cnn_model,img_batch)



	#Get second step
	predicted_series = trayectory.runSNN(snn_model,initial_step)
	print('The predicted trayectory is: \n', predicted_series)

	aux.plotList(predicted_series,'Time step','State')

