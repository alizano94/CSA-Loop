import os
import tensorflow as tf
import numpy as np


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import plotly.graph_objects as go


from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Utils.Helpers import Helpers


class CNN():

	def __init__(self):
		pass

	def createCNN(self,IMG_H=212,IMG_W=212,chan=1,summary=False):
		'''
		function that creates and compile the CNN
		args:
			-IMG_H:Images height (def 212px)
			-IMH_W:Images width (def 212px)
			-chan: color channels (def 1 i.e. grayscale)
		returns:
			-model: keras model.
		'''
		# Model Creation
		model = Sequential()
		model.add(Conv2D(16, 3, padding='same', activation='relu',
			input_shape=(IMG_H, IMG_W, chan)))
		model.add(MaxPooling2D())
		model.add(Dropout(0.2))
		model.add(Conv2D(32, 3, padding='same', activation='relu'))
		model.add(MaxPooling2D())
		model.add(Conv2D(64, 3, padding='same', activation='relu'))
		model.add(MaxPooling2D())
		model.add(Dropout(0.2))
		model.add(Conv2D(128, 3, padding='same', activation='relu'))
		model.add(MaxPooling2D())
		model.add(Dropout(0.2))
		model.add(Conv2D(256, 3, padding='same', activation='relu'))
		model.add(MaxPooling2D())
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dense(3, activation='softmax'))

		#Compile the model
		model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

		if summary:
			model.summary()
			tf.keras.utils.plot_model(
				model = model,
				rankdir="TB",
				dpi=72,
				show_shapes=True
				)

		return model 

	def trainCNN(self,PATH,model,batch=32,epochs=8,IMG_H=212,IMG_W=212):
		#Make Image size parameters global variables. 
		'''
		A function that trains a CNN given the model
		and the PATH of the data set.
		'''
		train_dir = os.path.join(PATH,'train')
		test_dir = os.path.join(PATH,'test')
		train_crystal_dir = os.path.join(train_dir,'0')
		train_fluid_dir = os.path.join(train_dir,'1')
		train_defective_dir = os.path.join(train_dir,'2')

		test_crystal_dir = os.path.join(test_dir,'0')
		test_fluid_dir = os.path.join(test_dir,'1')
		test_defective_dir = os.path.join(test_dir,'2')

		#Process the Data
		image_gen = ImageDataGenerator(rescale=1./255)
		train_data_gen = image_gen.flow_from_directory(
			#batch_size=batch,
			directory=train_dir,
			color_mode='grayscale',
			shuffle=True,
			target_size=(IMG_H, IMG_W),
			class_mode='categorical')
		#print(train_data_gen.shape)

		test_data_gen = image_gen.flow_from_directory(
			#batch_size=batch,
			directory=test_dir,
			color_mode='grayscale',
			target_size=(IMG_H, IMG_W),
			class_mode='categorical')

		

		num_crystal_train = len(os.listdir(train_crystal_dir))
		num_fluid_train = len(os.listdir(train_fluid_dir))
		num_defective_train = len(os.listdir(train_defective_dir))

		num_crystal_test = len(os.listdir(test_crystal_dir))
		num_fluid_test = len(os.listdir(test_fluid_dir))
		num_defective_test = len(os.listdir(test_defective_dir))

		total_train = num_crystal_train + num_fluid_train + num_defective_train
		total_test = num_crystal_test + num_fluid_test + num_defective_test
		
		history = model.fit(
			train_data_gen,
			steps_per_epoch=total_train // batch,
			epochs=epochs,
			validation_data=test_data_gen,
			validation_steps=total_test // batch,
			callbacks = [tf.keras.callbacks.EarlyStopping(
				monitor='val_loss',
				min_delta=0.01,
				patience=7
				)]
			)

		

		#Plot Accuracy, change this to matplotlib
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=history.epoch,
                         y=history.history['accuracy'],
                         mode='lines+markers',
                         name='Training accuracy'))
		fig.add_trace(go.Scatter(x=history.epoch,
                         y=history.history['val_accuracy'],
                         mode='lines+markers',
                         name='Validation accuracy'))
		fig.update_layout(title='Accuracy',
                  xaxis=dict(title='Epoch'),
                  yaxis=dict(title='Percentage'))
		fig.show()

	def testCNN(self,model,path):
		'''
		Function thats test the CNN against its own DS.
		Return class mtrix as numpy array
		'''
		aux = Helpers()
		out_file = open(path+'error+log.txt',"w+")
		out_file.write('#file,real_cat,pred_cat\n')
		sep = '###################'
		category = ['Fluid', 'Defective', 'Crystal']


		train_path = path+'/train/'
		test_path = path+'/test/'
		paths = [train_path,test_path]
		d = len(next(os.walk(train_path))[1])
		class_mat = np.zeros((d,d))

		for direct in paths:
			out_file.write(sep+direct+sep+'\n')
			for entry in os.scandir(direct):
				if entry.is_dir():
					cat = int(str(entry.path)[-1])
					out_file.write(sep+entry.path+sep+'\n')
					print(cat)
					for file in os.scandir(entry.path):
						if (file.path.endswith('.png')) and file.is_file():
							img_batch = aux.preProcessImg(file.path)
							pred, dum = self.runCNN(model,img_batch)
							class_mat[pred][cat] += 1
							if pred != cat:
								log = file.path+','+category[cat]+','+dum+'\n'
								out_file.write(log)
		out_file.close()
		return class_mat




	def runCNN(self,model,img_batch):
		category = ['Fluid', 'Defective', 'Crystal']
		prediction = model.predict(img_batch)
		cat_index = np.argmax(prediction[0])

		return cat_index, category[cat_index]