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
cnn_out_ds_path = './CNN/testDS'
step = 1

#Create the models
cnn_model = img_class.createCNN(summary=False)
snn_model = trayectory.createSNN(step,summary=False)


#Train the models
cnn_training_flag = False
snn_training_flag = True

if cnn_training_flag:
	img_class.trainCNN(cnn_out_ds_path,cnn_model,epochs=16)
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
cnn_test_ownDS = False
cnn_test_outDS = False


if cnn_test_flag:
	if cnn_test_ownDS:
		print(img_class.testCNN(cnn_model,cnn_ds_path))
	if cnn_test_outDS:
		cat_mat = img_class.testCNN(cnn_model,cnn_out_ds_path)
		print(cat_mat)
		print(np.sum(cat_mat))

#Run the loop
loop_flag = True

if loop_flag:
	#Run the CNN to get inital step
	img_path = './InitialStates/4V-5tray-99step10s.png'
	img_batch = aux.preProcessImg(img_path)

	initial_step, initial_step_label = img_class.runCNN(cnn_model,img_batch)



	#Get second step
	vol_lvl = 1
	time_stamp = 0
	predicted_series = trayectory.runSNN(snn_model,initial_step,
										vol_lvl,time_stamp,step,length=200)
	print('The predicted trayectory is: \n', predicted_series)

	aux.plotList(predicted_series,'Time step','State')

