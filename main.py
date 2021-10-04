import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import nbinom
import tensorflow as tf
import tensorflow_probability as tfp


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
snn_model = trayectory.createDNN(step,summary=True)
#snn_model = trayectory.createCSNN(step,summary=True)
#snn_model = trayectory.createDSNN(step,summary=True)


#Train the models
cnn_training_flag = False
snn_training_flag = False

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
	trayectory.trainSNN(snn_ds_path,snn_model,step,epochs=50,batch=5)
	#trayectory.trainSNNsingleFeat(snn_ds_path,snn_model,step,epochs=20)
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

snn_test_flag = False


if cnn_test_flag:
	if cnn_test_ownDS:
		print(img_class.testCNN(cnn_model,cnn_ds_path))
	if cnn_test_outDS:
		cat_mat = img_class.testCNN(cnn_model,cnn_out_ds_path)
		print(cat_mat)
		print(np.sum(cat_mat))

if snn_test_flag:
	print('Calculating one step transition probabilities...........')
	for v in [1,2,3,4]:
		for init in [0,1,2]:
			example_dict = {'Si': np.array([init]),
							'V':np.array([v])
							}

			pred_dist = trayectory.runSNN(snn_model,example_dict)
			print('############Input feature################')
			print(example_dict)
			print(pred_dist)
			print(sum(pred_dist[0]))

###########Generate transition probabilities############################
trans_prob_DNN = False

if trans_prob_DNN:
	n = 500
	bars = ['Fluid','Defective','Crystal']
	x_pos = np.arange(len(bars))
	plt.yticks(color='black')
	fig_path = './Results/DNN/100s/MVR/Drop/'
	print('Calculating '+str(n)+' step transition probabilities...........')
	for v in [1,2,3,4]:
		for init in [0,1,2]:
			plt.xticks(x_pos, bars, color='black')
			fig_name = fig_path+'MVRDS-L-S'+str(init)+'-V'+str(v)+'.png'
			example_dict = {'Si': np.array([init]),
							'V':np.array([v])}
			print(example_dict)
			probs = trayectory.runSNN(snn_model,example_dict)
			print(probs)
			cat_dist = tfp.distributions.Categorical(probs=probs[0])
			empirical_prob = tf.cast(
					tf.histogram_fixed_width(
						cat_dist.sample(int(n)),
										[0, 2],
										nbins=3
										),dtype=tf.float32) / n
			print(empirical_prob)
			plt.bar(x_pos,empirical_prob,color='black')
			plt.savefig(fig_name)
			plt.clf()

###########Generate transition probabilities############################
trans_prob_SNN = False

if trans_prob_SNN:
	n = 500
	bars = ['Fluid','Defective','Crystal']
	x_pos = np.arange(len(bars))
	plt.yticks(color='black')
	fig_path = './Results/DSNN/100s/MVR/Drop'
	print('Calculating '+str(n)+' step transition probabilities...........')
	for v in [1,2,3,4]:
		for init in [0,1,2]:
			empirical_prob = [0,0,0]
			plt.xticks(x_pos, bars, color='black')
			fig_name = fig_path+'MVRDS-L-S'+str(init)+'-V'+str(v)+'.png'
			example_dict = {'Si': np.array([init]),
							'V':np.array([v])}
			print(example_dict)
			for i in range(n):
				probs = trayectory.runSNN(snn_model,example_dict)
				state = np.argmax(probs[0])
				empirical_prob[state] += 1
			print(empirical_prob)
			plt.bar(x_pos,empirical_prob,color='black')
			plt.savefig(fig_name)
			plt.clf()
			




#Main Code: add any functionalities here.
#Run the CNN to get inital step
img_path = './InitialStates/test.png'
#img_path = './InitialStates/4V-5tray-99step10s.png'
img_batch = aux.preProcessImg(img_path)

S0, _ = img_class.runCNN(cnn_model,img_batch)
v = 4

init_feat = {'Si': np.array([S0]),
			 'V':np.array([v])}

pred_traj = trayectory.trajectory(snn_model,init_feat,10)

print(pred_traj)

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


###########################Miscelanious####################################
