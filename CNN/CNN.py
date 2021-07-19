import os
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import plotly.graph_objects as go


#Load the Data Set
PATH = './DS' #path from where to load the DS

train_dir = os.path.join(PATH,'train')
test_dir = os.path.join(PATH,'test')

train_crystal_dir = os.path.join(train_dir,'0')
train_fluid_dir = os.path.join(train_dir,'1')
train_defective_dir = os.path.join(train_dir,'2')

test_crystal_dir = os.path.join(test_dir,'0')
test_fluid_dir = os.path.join(test_dir,'1')
test_defective_dir = os.path.join(test_dir,'2')

print('Length of training set', len(os.listdir(train_crystal_dir)))
print('Length of training set', len(os.listdir(test_crystal_dir)))

#Process the Data
IMG_HEIGHT = 212
IMG_WIDTH = 212
batch_size = 32

image_gen = ImageDataGenerator(rescale=1./255)

train_data_gen = image_gen.flow_from_directory(
    #batch_size=batch_size,
    directory=train_dir,
    color_mode='grayscale',
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

test_data_gen = image_gen.flow_from_directory(
    #batch_size=batch_size,
    directory=test_dir,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')


# Model Creation
model = Sequential()
model.add(Conv2D(16, 3, padding='same', activation='relu',
	input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
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
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))


#Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()

#Model Training
batch_size = 32
epochs = 8

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
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=test_data_gen,
    validation_steps=total_test // batch_size,
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.01,
        patience=7
    )]
)

#Plot Accuracy
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

#Save the model
save_path = '../SavedModels/'

model.save_weights(os.path.join(save_path, 'CSA.h5'))

model_json = model.to_json()

with open(os.path.join(save_path, 'CSA.json'), 'w') as json_file:
  json_file.write(model_json)

json_file.close()