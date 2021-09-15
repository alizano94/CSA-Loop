#############################################################################################

import numpy as np
import matplotlib.pyplot as plt

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

############################DS Histogram#################################

PATH = './SNN/DS'

train_features = pd.DataFrame()
train_labels = pd.DataFrame()

for file in os.listdir(PATH):
	if file.endswith('.csv'):
		train_csv = PATH+'/'+file
		X, Y = aux.preProcessTens(train_csv)
		train_features = train_features.append(X)
		train_labels = train_labels.append(Y)

print(train_features)
print(train_labels)

train_features = np.array(train_features,dtype=float)
train_labels = np.array(train_labels,dtype=float)

labels = ['Fluid','Defective','Crystal']
values = [[0,0,0],
		[0,0,0],
		[0,0,0],
		[0,0,0]]


for i in range(len(train_labels)):
	j = int(train_features[i][1]-1)
	if j == -1:
		j += 1
	k = train_labels[i][0]
	values[j][int(k)] += 1

	S = [0,0,0]
	for i in [0,1,2]:
		for j in [0,1,2,3]:
			S[i] = S[i] + values[j][i]
	if S[0] != S[2] or S[1] != S[2]: