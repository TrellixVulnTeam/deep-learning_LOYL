'''
    Neural Network Classifier for MNIST
'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import scipy
import sklearn
import matplotlib


img_dim = 28
num_classes = 10

def img_vec(x, num_examples, rgb_scale = 255.):
    x = x.reshape(num_examples, img_dim ** 2)
    x = x.astype('float32')
    return x / rgb_scale


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = img_vec(x_train, 60000)
x_test = img_vec(x_test, 10000)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#
# Feed-forward model
#
model = Sequential([
    Dense(32, input_dim=(img_dim ** 2)),
    Activation('relu'),
    Dense(16),
    Activation('relu'),
    Dense(3),
    Activation('relu'),
    Dense(num_classes),
    Activation('softmax'),
])


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train)

score = model.evaluate(x_test, y_test)
print('Model results:')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
