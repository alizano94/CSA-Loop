import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

train_DS = pd.read_csv(
    "./VRand-10s-T5.csv")
train_features = train_DS.copy()
train_features.drop(train_features.tail(1).index,inplace=True)
train_features.drop(columns=['cat','#time'],inplace=True)

train_labels = train_DS.copy()
train_labels.drop(columns=['cat','V_level','#time'],inplace=True)
train_labels.drop(train_labels.head(1).index,inplace=True)
train_labels =	np.array(train_labels)
y_data = np.zeros((len(train_labels),3),dtype=int)
for i in range(len(train_labels)):
	y_data[i,train_labels[i][0]] = 1 

inputs = {}

for name, column in train_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

print(inputs)

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

print(list(numeric_inputs.values()))

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(train_DS[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

print(all_numeric_inputs)

preprocessed_inputs = [all_numeric_inputs]

print(preprocessed_inputs)

for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = preprocessing.StringLookup(vocabulary=np.unique(train_features[name]))
  one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

train_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

tf.keras.utils.plot_model(model = train_preprocessing , rankdir="TB", dpi=72, show_shapes=True)

train_features_dict = {name: np.array(value)
						for name, value in train_features.items()}

print(train_features_dict)

def titanic_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(16),
    tfp.layers.DenseFlipout(32),
    layers.Dense(3, activation='softmax'),
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss='categorical_crossentropy',
                optimizer=tf.optimizers.Adam())
  return model
print(train_features_dict)
print(y_data)
model = titanic_model(train_preprocessing, inputs)
model.summary()

model.fit(x=train_features_dict, y=y_data, epochs=100)

print("##################Prediction####################")
example_dict = {'cat_index': np.array([2]),
				'V_level':np.array([4])}

predict = model.predict(example_dict)
print(example_dict)
print(predict)
print(np.argmax(predict))
