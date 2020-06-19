import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

NAME = 'Cats-vs-Dogs-CNN'

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

def unpickle_this(filename):
    with open(filename, 'rb') as file:
        unpickled = pickle.load(file)
    return unpickled



y = np.array(unpickle_this('/storage/y.pickle'))

X = unpickle_this('/storage/X.pickle')
X = X / 255.0
type(X)

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y,
          batch_size=32,
          epochs=1,
          validation_split=.3)