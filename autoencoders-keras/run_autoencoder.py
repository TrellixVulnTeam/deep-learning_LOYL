import autoencoder
from keras import regularizers
from keras.models import Model, load_model
import numpy as np


def main():
  autoencoder1 = autoencoder.Autoencoder(
    layer_dims=[784, 128, 36, 128, 784],
    epochs=1,
    activity_regularizer=None, #regularizers.l1(0),
    use_pretrained=False,
    model_name="128_36") ### TODO: CHANGE THE NAME!!!!!!!!!

  m = load_model("128_36_autoencoder.h5")
  # m.compile(optimizer='adadelta', loss='binary_crossentropy')
  # print(m.layers[0].get_weights())
  for layer in m.layers:
    weights = layer.get_weights()

  for i in range(len(weights)):
    print(i)
    print(weights[i].shape)

  print(type(weights[0]))

  # W = []
  # for i, layer in enumerate(m.layers):
  #   W[i] = (m.layers)[i].get_weights()

  # x_train, y_train, x_test, y_test = autoencoder1.mnist_data()
  # x_train, y_train, x_test, y_test = x_train[:4000], y_train[:4000], x_test[:4000], y_test[:4000]

  # print(np.shape(x_train))

  # autoencoder1.visualize()


if __name__ == "__main__":
  main()