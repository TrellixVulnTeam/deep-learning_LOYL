# From: https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.datasets import mnist
from keras.models import load_model
import numpy as np
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



#
# Fully connected Neural Layer as encoder and decoder:
#
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
encoding_dim1 = 16

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
encoded = Dense(encoding_dim1, activation='relu')(encoded)


# "decoded" is the lossy reconstruction of the input
decoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

# For later: decoder,encoder
# #
# # Seperate Encoder Model:
# #
# # this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# #
# # Seperate Decoder Model:
# #
# # create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim1,))
# # retrieve the last layer of the autoencoder model
decoder_layer1= autoencoder.layers[-2]
decoder_layer2 = autoencoder.layers[-1]
# # create the decoder model
decoder = Model(encoded_input, decoder_layer1(encoded_input))
decoder = Model(decoder.output, decoder_layer2(32))

# Autoencoder
# this model maps an input to its reconstruction

# Binary crossentropy loss is per-pixel
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# # encode and decode some digits
# # note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# # t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
# tsne = TSNE(n_components=2, random_state=0)
# x_test_2d = tsne.fit_transform(x_test)

# plt.figure()
# for idx, cl in enumerate(np.unique(x_test)):
#     plt.scatter(x=x_test_2d[x_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
# plt.xlabel('X in t-SNE')
# plt.ylabel('Y in t-SNE')
# plt.legend(loc='upper left')
# plt.title('t-SNE visualization of test data')
# plt.show()



# def main():
#     try:
#         load_model('my_model.h5')
#     except:
#         create_autoencoder()

#     visualize_autoencoder()
#     model.save('my_model.h5')

# if __name__ == '__main__':
#     main()

'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import scipy
import sklearn
import matplotlib

#
# Auto Encoder
#
auto_encoder = Sequential([
    Dense(32, input_dim=(img_dim ** 2)),
    Activation('relu'),
    Dense(16),
    Activation('relu'),
    Dense(3),
    Activation('relu'),
    Dense(3),
    Activation('relu'),
    Dense(16),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(img_dim ** 2),
    
])

auto_encoder.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = auto_encoder.fit(x_train, x_train)

score = auto_encoder.evaluate(x_test, x_test)
print('Model results:')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

'''
