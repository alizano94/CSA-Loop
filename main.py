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
cnn_model = img_class.createCNN(summary=True)
snn_model = trayectory.createSNN(step,summary=True)


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
	print(initial_step)


	#Get second step
	vol_lvl = 4
	t_step = 10
	time_stamp = 0
	length = 100

	inp = [[float(initial_step),
				time_stamp,
				vol_lvl]]
	inp = np.transpose([inp[-1]] * step)
	inp = np.reshape(inp,(1,3,step))
	x = [inp]
	out =[inp[0][0][0]]
	for i in range(length):
		pred = trayectory.runSNN(snn_model,inp)
		print(pred)

		# Find index of maximum value from 2D numpy array
		result = np.where(pred == np.amax(pred))
		# zip the 2 arrays to get the exact coordinates
		listOfCordinates = list(zip(result[0], result[1]))
		index = listOfCordinates[0][1]
		out.append(index)
		init = out[-1]
		time_stamp += t_step
		new = [[init,time_stamp,vol_lvl]]
		inp = np.append(inp,np.reshape(new,(1,3,1)),axis=2)
		inp = np.delete(inp,0,axis=2)
		inp = np.reshape(inp,(1,3,step))
		x.append(inp)

	print('The predicted trayectory is: \n', out)
	#print('For the inputs: \n', x)

	aux.plotList(out,'Time step','State')

