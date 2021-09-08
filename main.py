import os
import numpy as np
import matplotlib.pyplot as plt

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
snn_model = trayectory.createBBNN(step,summary=False)
#snn_model = trayectory.createPBNN(step,summary=True)


#Train the models
cnn_training_flag = False
snn_training_flag = False

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
	trayectory.trainSNN(snn_ds_path,snn_model,step,epochs=100)
	aux.saveWeights(snn_model,save_model_path,snn_weights_name)

#Load weigths
cnn_load_flag = False
snn_load_flag = False

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

#Main Code: add any functionalities here.
loop_flag = True

if loop_flag:
	#Run the CNN to get inital step
	#img_path = './InitialStates/test.png'
	#img_batch = aux.preProcessImg(img_path)

	#initial_step, initial_step_label = img_class.runCNN(cnn_model,img_batch)
	#print(initial_step)

	snn_ds_path = './SNN/test-DS/V1'
	#snn_ds_path = './SNN/DS'
	#Train the models
	snn_training_flag = True

	if snn_training_flag:
		weigth_file = save_model_path+snn_weights_name+'.h5'
		if os.path.isfile(weigth_file):
			os.remove(weigth_file)
		trayectory.trainSNN(snn_ds_path,snn_model,step,epochs=100)
		aux.saveWeights(snn_model,save_model_path,snn_weights_name)

	#Load weigths
	snn_load_flag = True

	if snn_load_flag:
		print('Loading SNN model...')
		load_path_snn = save_model_path+snn_weights_name+'.h5'
		aux.loadWeights(load_path_snn,snn_model)

	#Get second step
	X = range(10)
	for j in range(4):
		add = 1
		vol_lvl = j+add
		for k in range(3):
			example_dict = {'cat_index': np.array([k]),
							'V_level':np.array([vol_lvl])}

			for i in range(10):
				pred_params = trayectory.runSNN(snn_model,example_dict)
				n = pred_params[:,0]; p = pred_params[:,1]
				plt.plot(X, nbinom.pmf(X, n, p), 'o', ms=8)
			plt.xlabel('State')
			#plt.title('Binomial probabilities. \nDS: VR\nPrior State: 0\nV_lvl: 4')
			plt.ylim(bottom=0)
			plt.xlim([0.0,10.0])
			fig_name = './Results/Binoms/V'+str(add)+'DS-NTS-V'+str(vol_lvl)+'-IS'+str(k)+'.png'
			#fig_name = './Results/Binoms/VRDS-NTS-V'+str(vol_lvl)+'-IS'+str(k)+'.png'
			plt.savefig(fig_name)
			plt.clf()
			#plt.show()
