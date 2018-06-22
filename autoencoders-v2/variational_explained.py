'''This script demonstrates how to build a variational autoencoder with Keras.
 #Reference
 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

import pdb
import matplotlib.pyplot as plt
# pdb.set_trace()

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

##
## Assumption: the true (but intractable) posterior, pθ(x, z), takes on
## an approximate Gaussian form with an approximately diagonal covariance.
##
## In this case, we can let the variational approximate posterior, qφ(z|x), be
## a multivariate Gaussian with a diagonal covariance structure:
##      log qφ(z|x(i)) = log N (z; μ(i), σ2(i)I)
##

##
## Sample from the posterior z(i,l) ∼ qφ(z|x(i))
def sampling(args):
    z_mean, z_log_var = args

    ## ???
    ## prior over the latent variables
    ## centered isotropic, Σ = (σ^2)I, multivariate Gaussian pθ(z) = N(z;0,I)
    ## Note: Lacks parameters
    ##
    ## ε is an auxiliary ε(l) ∼ N (0, I)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    ##

    ## z(i,l) = gφ(x(i), ε(l)) = μ(i) + σ(i) ⊙ ε(l)
    return z_mean + K.exp(z_log_var / 2) * epsilon

##
## Encoder neural network for the probabilistic encoder for qφ(z|x)
## qφ(z|x) is the approximation to the posterior of the actual model that
## generates the data, pθ(x, z)
##
##
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
##
## ??? - How do these values become the mean and the variance, by  the loss
##  function and epsilon?
## μ and σ, are outputs of the encoding
## MLP, i.e. nonlinear functions of datapoint x(i) and the variational
## parameters φ 
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
##


# Decoder
# we instantiate these layers separately so as to reuse them later
#
## ??? - What does it mean for a NN to be of a particular distribution
## the decoding term log pθ (x(i) |z(i,l) ) can be an MLP that's:
## - Gaussian (in case of real-valued data)
## - Bernoulli (in case of binary data)
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)
##


# instantiate VAE model
vae = Model(x, x_decoded_mean)

##
## Loss
## 1 + log((σ(i))^2) − (μ(i))^2 − (σ(i))^2
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), 
    axis=-1)
# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)

## L(θ,φ;x(i))
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

plt.imshow(x_train[0].reshape(28, 28))
plt.show()

exit()

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()



# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
