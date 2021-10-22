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
save_model_path = './SavedModels/'
cnn_weights_name = 'CNN'
snn_weights_name = 'SNN'
cnn_out_ds_path = './CNN/testDS'
step = 4

#Create the models
cnn_model = img_class.createCNN(summary=False)
#snn_model = trayectory.createDNN(step,summary=True)
snn_model = trayectory.createRNN(step,summary=True)
#snn_model = trayectory.createCSNN(step,summary=True)
#snn_model = trayectory.createDSNN(step,summary=True)


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
	trayectory.trainModel(snn_ds_path,snn_model,step,epochs=100,batch=4)
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
	fig_path = './Results/LSTM/'+str(step)+'stepMem/'
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
			#plt.bar(x_pos,empirical_prob,color='black')
			plt.bar(x_pos,probs[0],color='black')
			plt.savefig(fig_name)
			plt.clf()

###########Generate transition probabilities############################
trans_prob_LSTM = False

if trans_prob_LSTM:
	n = 500
	bars = ['Fluid','Defective','Crystal']
	x_pos = np.arange(len(bars))
	plt.yticks(color='black')
	fig_path = './Results/LSTM/'+str(step)+'stepMem/'
	print('Calculating '+str(n)+' step transition probabilities...........')
	for v in [1,2,3,4]:
		for init in [0,1,2]:
			plt.xticks(x_pos, bars, color='black')
			fig_name = fig_path+'MVRDS-L-S'+str(init)+'-V'+str(v)+'.png'

			example_dict = {'V':np.array([v])}
			for i in range(step):
				name = 'S'+str(i-step)
				example_dict[name] = np.array([init])
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
			#plt.bar(x_pos,probs[0],color='black')
			plt.savefig(fig_name)
			plt.clf()

###########Generate transition probabilities############################
trans_prob_SNN = False

if trans_prob_SNN:
	n = 500
	bars = ['Fluid','Defective','Crystal']
	x_pos = np.arange(len(bars))
	plt.yticks(color='black')
	fig_path = './Results/LSTM/'
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

loop_flag = True
if loop_flag:
	#Run the CNN to get inital step
	img_path = './InitialStates/test.png'
	#img_path = './InitialStates/4V-5tray-99step10s.png'
	img_batch = aux.preProcessImg(img_path)

	length = 10
	x = np.arange(length+1)
	bars = ['Fluid','Defective','Crystal']
	y_pos = np.arange(len(bars))
	fig_path = './'



	for v in [1,2,3,4]:
		for S0 in [0,1,2]:
			#S0, _ = img_class.runCNN(cnn_model,img_batch)
			init_feat = {'V':np.array([v])}
			for i in range(step):
				name = 'S'+str(i-step)
				init_feat[name] = np.array([S0])
			
			fig_name = 'LSTM-PredTraj-V'+str(v)+'-S0-'+str(S0)+'-10steps.png'

			pred_traj, v_traj  = trayectory.trajectory(step,snn_model,init_feat,length)
			print(init_feat)
			print([S0]+pred_traj)
			y = np.array([S0]+pred_traj)
			v_traj = np.array([v]+v_traj)

			fig, ax = plt.subplots()
			twin1 = ax.twinx()

			p1 = ax.scatter(x,y,label="Output State")
			p2, = twin1.plot(x,v_traj,"r-", label="Voltage Level")

			ax.set_ylim(-0.1,2.1)
			twin1.set_ylim(-0.1, 4.1)
			ax.set_yticks(y_pos)
			ax.set_yticklabels(bars)

			ax.set_xlabel("Time Step")
			twin1.set_ylabel("Voltage Level")

			tkw = dict(size=4, width=1.5)
			ax.tick_params(axis='y', **tkw)
			twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
			ax.tick_params(axis='x', **tkw)

			ax.legend(handles=[p1, p2])

			plt.savefig(fig_path+fig_name)
			plt.clf()





###########################Miscelanious####################################

#Main Code: add any functionalities here.

loop_flag = False
if loop_flag:

	length = 100
	x = np.arange(length+1)
	bars = ['Fluid','Defective','Crystal']
	y_pos = np.arange(len(bars))
	fig_path = './'

	for v in [1,2,3,4]:
		for S0 in [0,1,2]:
			traj = [x]
			for i in range(20):
				#S0, _ = img_class.runCNN(cnn_model,img_batch)
				init_feat = {'V':np.array([v])}
				for i in range(step):
					name = 'S'+str(i-step)
					init_feat[name] = np.array([S0])
				
				fig_name = 'LSTM-PredTraj-V'+str(v)+'-S0-'+str(S0)+'-10steps.png'

				pred_traj, v_traj  = trayectory.trajectory(step,snn_model,init_feat,length)
				print(init_feat)
				print([S0]+pred_traj)
				y = np.array([S0]+pred_traj)
				traj = np.vstack((traj,y))
				v_traj = np.array([v]+v_traj)

			fig, ax = plt.subplots()
			twin1 = ax.twinx()
			for i in range(19):
				p1 = ax.plot(traj[0,:],traj[i+1,:],label="Output State")
				p2, = twin1.plot(x,v_traj,"r-", label="Voltage Level")

				ax.set_ylim(-0.1,2.1)
				twin1.set_ylim(-0.1, 4.1)
				ax.set_yticks(y_pos)
				ax.set_yticklabels(bars)

				ax.set_xlabel("Time Step")
				twin1.set_ylabel("Voltage Level")

				tkw = dict(size=4, width=1.5)
				ax.tick_params(axis='y', **tkw)
				twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
				ax.tick_params(axis='x', **tkw)

				

			plt.savefig(fig_path+fig_name)
			plt.clf()





###########################Miscelanious####################################
##Obtain probability tensor###
prob_tens = np.empty([4,3,3])
for v in [1,2,3,4]:
	for S0 in [0,1,2]:
		init_feat = {'V':np.array([v])}
		for i in range(step):
			name = 'S'+str(i-step)
			init_feat[name] = np.array([S0])
		#print(init_feat)
		probs = trayectory.runSNN(snn_model,init_feat)
		print(sum(probs[0]))
		prob_tens[v-1,S0,:] = probs[0]
print(prob_tens)



