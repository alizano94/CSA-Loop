import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import nbinom

from Utils.Helpers import Helpers
from CNN.CNN import *
from SNN.SNN import *

#Load necesary modules
aux = Helpers()
img_class = CNN()
trayectory = SNN()

#Define variables
cnn_ds_path = './CNN/DS'
snn_ds_path = './SNN/test-DS/MVR'
#snn_ds_path = './SNN/DS'
save_model_path = './SavedModels/'
cnn_weights_name = 'CNN'
snn_weights_name = 'SNN'
cnn_out_ds_path = './CNN/testDS'
step = 1

#Create the models
cnn_model = img_class.createCNN(summary=False)
#snn_model = trayectory.createBBNN(step,summary=False)
snn_model = trayectory.createCSNN(step,summary=True)


#Train the models
cnn_training_flag = False
snn_training_flag = True

if cnn_training_flag:
	weigth_file = save_model_path+cnn_weights_name+'.h5'
	if os.path.isfile(weigth_file):
		os.remove(weigth_file)
	img_class.trainCNN(cnn_out_ds_path,cnn_model,epochs=100)
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

#Testing part.
#Test NN
cnn_test_flag = False
cnn_test_ownDS = True
cnn_test_outDS = True

snn_test_flag = True


if cnn_test_flag:
	if cnn_test_ownDS:
		print(img_class.testCNN(cnn_model,cnn_ds_path))
	if cnn_test_outDS:
		cat_mat = img_class.testCNN(cnn_model,cnn_out_ds_path)
		print(cat_mat)
		print(np.sum(cat_mat))

if snn_test_flag:
	for v in [1,2,3,4]:
		for init in [0,1,2]:
			initial_state = init
			vol_lvl = v
			example_dict = {'Si': np.array([initial_state]),
							'V':np.array([vol_lvl])
							}

			pred_dist = trayectory.runSNN(snn_model,example_dict)
			print('############Input feature################')
			print(example_dict)
			print(pred_dist)
			print(sum(pred_dist[0]))


#Main Code: add any functionalities here.
loop_flag = False

if loop_flag:
	vol_lvl = 4
	t_step = 100

	#Run the CNN to get inital step
	img_path = './InitialStates/test.png'
	img_batch = aux.preProcessImg(img_path)

	S0, S0_cat_label = img_class.runCNN(cnn_model,img_batch)

	init_feat = {'Si': np.array([S0]),
				 'V':np.array([vol_lvl]),
				 '#time':np.array([0])}

	trajectory = [S0]
	
	for i in range(10):
		#print(init_feat)
		pred_dist = trayectory.runSNN(snn_model,init_feat)
		out_state = np.argmax(pred_dist[0])
		init_feat['Si'] = np.array([out_state])
		init_feat['#time'] = init_feat['#time'] + np.array([t_step])
		trajectory.append(out_state)

	out_traj = {'step':np.arange(11),
				'So':np.array(trajectory)}
	out_traj = pd.DataFrame(out_traj)
	out_traj.to_csv('./out.csv', sep='\t')




	