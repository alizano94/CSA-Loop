import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from IPython.display import clear_output


from Utils.Helpers import *
from CNN.CNN import *
from SNN.SNN import *
from RL.RL import *

#load classes
Helpers = Helpers()
CNN = CNN()
SNN = SNN()
RL = RL()


#Define Variables
#FLAGS
cnn_train = False
snn_train = False
rl_train = False
preprocess_snnDS = False
Test_CNN = True
Test_SNN = False
feed_CNN = False
run_loop = False
constant_V = False


#Parameters
k = 3
memory = 1
window = 100

#paths
cnn_ds_dir = './CNN/DS/'
snn_ds_dir = './SNN/DS/'
#snn_ds_dir = '/home/lizano/Documents/CSA-Loop/SNN/DS/'
rl_ds_dir = './RL/'
csv_snnDS_path = 'Balanced-W'+str(window)+'-M'+str(memory)+'.csv'
csv_snnDS_path = snn_ds_dir+csv_snnDS_path
weights_dir = './SavedModels/'
cnn_weights = weights_dir+'CNN.h5'
snn_weights = weights_dir+'SNN'+str(window)+'-M'+str(memory)+'.h5'
q_table_file = rl_ds_dir+str(k**(memory))+'X4Q_table'+str(memory)+'M.npy'

#CNN
#Create CNN model
cnn_model = CNN.createCNN(summary=True)
#Tain the model or load learning
if cnn_train:
	if os.path.isfile(cnn_weights):
		os.remove(cnn_weights)
	CNN.trainCNN(cnn_ds_dir,cnn_model,epochs=100)
	#Check arguments for this method
	Helpers.saveWeights(cnn_model,cnn_weights)
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
	Helpers.saveWeights(snn_model,snn_weights)
else:
	print('Loading SNN model...')
	Helpers.loadWeights(snn_weights,snn_model)


#RL
if rl_train:
	if os.path.isfile(q_table_file):
		os.remove(q_table_file)
	q_table = RL.get_Q_table(snn_model,memory,k)
else:
	pass
	print('Loading Q table...')
	q_table = np.load(q_table_file)
print(q_table)

#Test Data 
#Test CNN accuracy.
if Test_CNN:
	score = CNN.testCNN(snn_ds_dir,cnn_model,DS = 'CNN')
	print('Prediction accuracy for CNN :'+str(score))
if feed_CNN:
	dump_path = '/home/lizano/Documents/CSA-Loop/CNN/DS/Dump'
	CNN.feedSNN2CNN(snn_ds_dir,dump_path)

if Test_SNN:
	Helpers.DataTrasnProbPlot(window,memory,k)
	SNN.testSNN(snn_model,window,memory,k)




#Control Loop
if run_loop:
	bars = ['Fluid','Defective','Crystal']
	y_pos = np.arange(len(bars))
	fig_path = './'
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.set_yticks(y_pos)
	ax1.set_yticklabels(bars)
	ax1.set_xlabel("Time Step")

	img_path = './InitialStates/test.png'
	img_batch = Helpers.preProcessImg(img_path)
	state, _ = CNN.runCNN(cnn_model,img_batch)

	trajectory = [state]
	policy = [0]

	while state != 2:
		action = RL.QControl(q_table,state)
		inp_feat = {'V':np.array([float(action)])}
		for j in range(memory):
			name = 'S'+str(j-memory)
			inp_feat[name] = np.array([float(state)])

		state = int(SNN.runSNN2(snn_model,inp_feat))
		trajectory.append(state)
		policy.append(action)

	print(trajectory)
	print(policy)

	length = len(trajectory)
	x = range(length)
	ax1.plot(x, trajectory,'--o')
	plt.savefig(fig_path+'out.png')
	plt.clf()




#Constat Trajecotry

if constant_V:
	length = 10
	x = range(length+1)


	bars = ['Fluid','Defective','Crystal']
	y_pos = np.arange(len(bars))
	fig_path = './Results/SNN/plots/trajecotries/'
	for V in [1,2,3,4]:
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1.set_yticks(y_pos)
		ax1.set_yticklabels(bars)
		ax1.set_xlabel("Time Step")
		for s in [0,1,2]:
			fig_name = 'PredTraj-V'+str(V)+'-S0-'+str(s)+'-10steps.png'
			init = {'V':np.array([V])}
			for i in range(memory):
				name = 'S'+str(i-memory)
				init[name] = np.array([s])
			print(init)
			path, pol = SNN.trajectory(memory,snn_model,init,length)
			print(path)
			ax1.plot(x, path,'--o')
		plt.savefig(fig_path+fig_name)
		plt.clf()



length = 10
x = range(length+1)


bars = ['Fluid','Defective','Crystal']
y_pos = np.arange(len(bars))
fig_path = './'
for V in [1,2,3,4]:
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.set_yticks(y_pos)
	ax1.set_yticklabels(bars)
	ax1.set_xlabel("Time Step")
	for s in [0,1,2]:
		fig_name = 'PredTraj-V'+str(V)+'-S0-'+str(s)+'-10steps.png'
		init = {'V':np.array([V])}
		for i in range(memory):
			name = 'S'+str(i-memory)
			init[name] = np.array([s])
		print(init)
		path, pol = SNN.trajectory(memory,snn_model,init,length)
		print(path)
		ax1.plot(x, path,'--o')
	plt.savefig(fig_path+fig_name)
	plt.clf()
