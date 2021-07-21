import os
from Helpers import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '../SNN/DS/' #path to SNN DS

h = Helpers()

ax = plt.gca()
trayectory = pd.DataFrame()

for i in range(3,6):
	for j in range(1,6):
		trayectory_file = str(i)+'V-'+str(j)+'tray.csv'
		col_name = trayectory_file.replace('.csv','')
		df = h.csv2df(path+trayectory_file)
		trayectory[col_name] = df[0]
		plt.figure()
		plt.xlabel('Time step')
		plt.ylabel('State')
		plt.ylim(-.5,2.5)
		plt.scatter(trayectory.reset_index()['index'],
			trayectory[col_name],
			s=0.5)
		plt.savefig(path+col_name+'.png')


print(trayectory)