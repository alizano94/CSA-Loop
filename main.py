import numpy as np
from Utils.Helpers import Helpers
from CNN.RunCNN import CNN

step = 2
x = np.arange(1,20)
load_path = './SavedModels/'

ds = Helpers()
img_class = CNN()

#dsX, dsY = ds.convertToMatrix(x,step)

#print("The x values are: \n", dsX)
#print("The y values are: \n", dsY)

img_path = './test.png'
img_batch = ds.preProcessImg(img_path)

model = img_class.loadCNN(load_path)
initial_step = img_class.RunCNN(model,img_batch)

print(initial_step)
