import os
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
#snn_model = trayectory.createSNN(step,summary=True)
snn_model = trayectory.createPBNN(step,summary=True)


#Train the models
cnn_training_flag = False
snn_training_flag = True

if cnn_training_flag:
	weigth_file = save_model_path+cnn_weights_name+'.h5'
	if os.path.isfile(weigth_file):
		os.remove(weigth_file)
	img_class.trainCNN(cnn_out_ds_path,cnn_model,epochs=16)
	aux.saveWeights(cnn_model,save_model_path,cnn_weights_name)

if snn_training_flag:
	weigth_file = save_model_path+snn_weights_name+'.h5'
	if os.path.isfile(weigth_file):
		os.remove(weigth_file)
	trayectory.trainSNN(snn_ds_path,snn_model,step,epochs=20)
	aux.saveWeights(snn_model,save_model_path,snn_weights_name)

#Load weigths
cnn_load_flag = False
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
cnn_test_ownDS = True
cnn_test_outDS = True


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
	img_path = './InitialStates/test.png'
	img_batch = aux.preProcessImg(img_path)

	initial_step, initial_step_label = img_class.runCNN(cnn_model,img_batch)
	#print(initial_step)


	#Get second step
	vol_lvl = 4
	t_step = 10
	time_stamp = 0
	length = 100
	example_dict = {'cat_index': np.array([initial_step]),
				'V_level':np.array([vol_lvl]),
				'#time':np.array([time_stamp])}

	for i in range(length):
		pred = trayectory.runSNN(snn_model,example_dict)
		print(pred.mean())
		pred = np.argmax(pred[0])
		example_dict = {'cat_index': np.array([pred]),
				'V_level':np.array([vol_lvl])}
		
	#print('The predicted trayectory is: \n', out)
	#print('For the inputs: \n', x)

	#aux.plotList(out,'Time step','State')

