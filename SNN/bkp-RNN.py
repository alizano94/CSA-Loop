import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import tensorflow_probability as tfp
import plotly.graph_objects as go

#Functions needed (refactor this)
def convertToMatrix(data, step):
 X, Y =[], []
 for i in range(len(data)-step):
  d=i+step  
  X.append(data[i:d,])
  Y.append(data[d,])
 return np.array(X), np.array(Y)


#Create a random data set
N = 1000    
Tp = 800    

csvname = "./DS/5V-1tray.csv"

results = []
with open(csvname) as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    headers = next(reader) 
    for row in reader: # each row is a list
        results.append(row)
x1 = []

for i in range(0,len(results)):
    x1.append(float(results[i][1]))

dfx1 = pd.DataFrame(x1)
dfx1.head()


csvname2 = "./DS/5V-2tray.csv"

results = []
with open(csvname) as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    headers = next(reader) 
    for row in reader: # each row is a list
        results.append(row)
x2 = []

for i in range(0,len(results)):
    x2.append(float(results[i][1]))

dfx2 = pd.DataFrame(x2)
dfx2.head()

#split the data sets
Xtrain=dfx1.values
Xtest=dfx2.values

#Reshape
step = 2
# add step elements into train and test
Xtest = np.append(Xtest,np.repeat(Xtest[-1,],step))
Xtrain = np.append(Xtrain,np.repeat(Xtrain[-1,],step))



# convert into dataset matrix
XtrainX,XtrainY =convertToMatrix(Xtrain,step)
XtestX,XtestY =convertToMatrix(Xtest,step)

print(len(XtrainX))

#Reshape to keras model
XtrainX = np.reshape(XtrainX, (XtrainX.shape[0], 1, XtrainX.shape[1]))
XtestX = np.reshape(XtestX, (XtestX.shape[0], 1, XtestX.shape[1]))

print(len(XtrainX))