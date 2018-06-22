from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras import regularizers

from sklearn.manifold import TSNE
import h5py
import numpy as np
import matplotlib.pyplot as plt

from functools import reduce
import os.path


class Autoencoder:
  """Autoencoder module.

  To see this model in TensorBoard run the following in the shell:
    tensorboard --logdir=/tmp/autoencoder

  """
  def __init__(self,
               layer_dims=[784, 32, 784],
               epochs=50,
               activity_regularizer=None,
               use_pretrained=False,
               model_name='autoencoder'):
    """Initializes the autoencoder.

    Args:
      layer_dims: dimensions of layers, first and last should be the same
    """
    self.layer_dims = layer_dims
    self.epochs = epochs
    self.activity_regularizer = activity_regularizer
    self.use_pretrained = use_pretrained
    self.model_name = model_name

    self.autoencoder, self.encoder, self.decoder = self._init_model()
    self.x_train, self.y_train, self.x_test, self.y_test = self.mnist_data()
    self._train()
    self.encoded_imgs = self.encoder.predict(self.x_test)
    self.decoded_imgs = self.decoder.predict(self.encoded_imgs)




  def _init_model(self):
    input_img = Input(shape=(self.layer_dims[0],))
    previous_layer = input_img
    hidden_activation = 'relu'
    final_activation = 'sigmoid'
    activation = hidden_activation
    decoder_start = len(self.layer_dims) // 2 + len(self.layer_dims) % 2
    print(decoder_start)

    for layer_dim in self.layer_dims[1:decoder_start]:
      print(layer_dim)
      encoded = Dense(layer_dim,
                      activation=activation,
                      activity_regularizer=
                        self.activity_regularizer)(previous_layer)
      previous_layer = encoded

    for layer_num, layer_dim in enumerate(self.layer_dims[decoder_start:]):
      print(layer_dim)
      if layer_num == len(self.layer_dims) - 1:
        activation = final_activation

      decoded = Dense(layer_dim,
                      activation=activation)(previous_layer)
      previous_layer = decoded


    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)

#Maybe not
    encoded_input = Input(shape=(self.layer_dims[decoder_start],))
    decoder_layers = [encoded_input] + autoencoder.layers[decoder_start:]
    decoded_decoder_only = reduce(lambda x, y: y(x), decoder_layers)
    decoder = Model(encoded_input, decoded_decoder_only)

    return autoencoder, encoder, decoder

  def mnist_data(self):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train)), np.prod(x_train.shape[1:]))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, y_train, x_test, y_test

  def _train(self):
    """Returns a trained the autoencoder that is saved"""
    autoencoder_filename = self.model_name + "_autoencoder.h5"
    encoder_filename = self.model_name + "_encoder.h5"
    decoder_filename = self.model_name + "_decoder.h5"

    if self.use_pretrained and os.path.isfile(autoencoder_filename):
      self.autoencoder = load_model(autoencoder_filename)
      self.encoder = load_model(encoder_filename)
      self.decoder = load_model(decoder_filename)
    else:
      self.autoencoder.compile(optimizer='adadelta',
                               loss='binary_crossentropy')
      self.autoencoder.fit(self.x_train, self.x_train,
                      epochs=self.epochs,
                      batch_size=256,
                      shuffle=True,
                      validation_data=(self.x_test, self.x_test),
                      callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

      self.autoencoder.save(autoencoder_filename)
      self.encoder.save(encoder_filename)
      self.decoder.save(decoder_filename)

  def visualize(self):
    n = 10
    fig1 = plt.figure(figsize=(20, 10))
    for i in range(n):
      self.visualize_digits(i, n)
      self.visualize_hidden_layers(i, n)

    fig2 = plt.figure(figsize=(6, 6))
    self.visualize_tsne()
    plt.show()

  def visualize_digits(self, i, n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(self.x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(self.decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  def visualize_hidden_layers(self, i, n):
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(self.encoded_imgs[i].reshape(6, 6)) #.T for 3D conv
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


  def visualize_tsne(self):
    plt.jet()
    encoded_imgs_embedded = TSNE(n_components=2).fit_transform(
                              self.encoded_imgs)
    plt.scatter(encoded_imgs_embedded[:, 0],
                encoded_imgs_embedded[:, 1],
                c=self.y_test)
    plt.colorbar()

