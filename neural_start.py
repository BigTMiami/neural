import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True,linewidth=np.nan, precision=2)

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train[0])

#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()

x_train = keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
x_test = keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

#print(x_train[0])

model = keras.models.Sequential()
#model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu, input_shape=x_train.shape[1:]))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(f'Loss: {val_loss:6.4f} Accuracy: {val_acc:6.4f}')

model_filename = 'first_testing.model'
model.save(model_filename)

reload_model = keras.models.load_model(model_filename)

predictions = reload_model.predict(x_test)
print(predictions[0])

print(np.argmax(predictions[0]))