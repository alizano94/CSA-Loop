import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow_probability as tfp
from tensorflow import keras
import tensorflow as tf


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		c = dataset[i:(i+look_back), :]
		dataX.append(c)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def prior(kernel_size, bias_size, dtype=None):
	n = kernel_size + bias_size
	prior_model = keras.Sequential(
		[
			tfp.layers.DistributionLambda(
				lambda t: tfp.distributions.MultivariateNormalDiag(
					loc=tf.zeros(n), scale_diag=tf.ones(n)
				)
			)
		]
	)
	return prior_model


def posterior(kernel_size, bias_size, dtype=None):
	n = kernel_size + bias_size
	posterior_model = keras.Sequential(
		[
			tfp.layers.VariableLayer(
				tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
			),
			tfp.layers.MultivariateNormalTriL(n),
		]
	)
	return posterior_model


# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
csv = './MVR/V4-10s-T5.csv'
dataframe = read_csv(csv, usecols=[1,3], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainY_HC = numpy.zeros((len(trainY),3),dtype=int)
testY_HC = numpy.zeros((len(trainY),3),dtype=int)

for i in range(len(trainY)):
	index = int(trainY[i])
	trainY_HC[i][index] = 1

for i in range(len(testY)):
	index = int(testY[i])
	testY_HC[i][index] = 1 

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0],trainX.shape[1],2))
testX = numpy.reshape(testX, (testX.shape[0],testX.shape[1],2))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back,2)))
model.add(Dense(2))
model.add(tfp.layers.DenseVariational(
	units=32,
	make_prior_fn=prior,
	make_posterior_fn=posterior,
	activation="sigmoid",))
model.add(Dense(3, activation='softmax'))

model.compile(
	loss = 'categorical_crossentropy',
	optimizer='adam'
	)

model.fit(trainX,
	trainY_HC,
	epochs=100,
	batch_size=1,
	verbose=2
	)


print('Calculating one step transition probabilities...........')
for v in [1,2,3,4]:
	for init in [0,1,2]:
		example_dict = {'Si': numpy.array([init]),'V':numpy.array([v])
						}
		pred_dist = model.predict(numpy.array([[[init,v]]]))
		print('############Input feature################')
		print(example_dict)
		print(pred_dist)
		print(sum(pred_dist[0]))


sample_size = 500
print('Calculating '+str(sample_size)+' step transition probabilities...........')
for v in [1,2,3,4]:
	for init in [0,1,2]:
		hist = [0,0,0]
		example_dict = {'Si': numpy.array([init]),
						'V':numpy.array([v])
						}
		for i in range(sample_size):
			pred_dist = model.predict(numpy.array([[[init,v]]]))
			hist[numpy.argmax(pred_dist)] +=1
		for i in range(3):
			hist[i] = hist[i]
		print(example_dict)
		print(hist)
		print(sum(hist))


