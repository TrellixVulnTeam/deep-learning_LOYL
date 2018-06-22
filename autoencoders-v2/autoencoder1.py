from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras import backend as K
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

np.random.seed(0)

## Create Model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(784))
model.add(Activation('sigmoid'))

model.summary()

model.compile(optimizer='adadelta', loss='mean_squared_error')

model.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
# Results:
# x_train.shape = (60000, 784)
# x_test.shape = (10000, 784)
##


### Get Representations of Each Layer
representations = [x_test]

for i in range(len(model.layers)):
    get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[i].output])
    representation = get_layer_output([x_test])[0]
    representations.append(representation)
# Results:
# representations is a list of each transformation through the neural net.
#   representations[0] -> input
#   representations[1] -> next non activated layer
#   representations[2] -> representations[1] but activated
#   ...
### 


#### Visualize Representations
num_examples = 10

plt.figure(figsize=(20, 10))
plt.gray()

for rep_num, rep in enumerate(representations):
    rep = rep[:num_examples]
    for i in range(num_examples):
        num_rows = len(representations)
        num_cols = num_examples
        curr_subplot = rep_num * num_examples + i + 1
        ax = plt.subplot(num_rows, num_cols, curr_subplot)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        size = rep[i].shape[0]
        if math.sqrt(size).is_integer():
            dim_size = int(math.sqrt(size))
            img_arr = rep[i].reshape(dim_size, dim_size)
        else:
            img_arr = rep[i].reshape(size, 1)

        plt.imshow(img_arr)
####


##### t-SNE Visualization
# arrs should be of shape: (number of examples, example size)
def visualize_tsne(arrs):
    plt.figure(figsize=(20, 10))
    plt.title("Original Dimension: ", arrs.shape[1])
    plt.jet()
    embedded = TSNE(n_components=2).fit_transform(arrs)
    plt.scatter(embedded[:, 0], embedded[:, 1], c=y_test)
    print(
    plt.colorbar()

visualize_tsne(representations[4])
#####


plt.show()


# Sequential model API
# Look at console for printed summary
# model.get_config() -> Dictionary containing the configuration of the model
# model.get_weights() -> Gets the weights of the model as numpy arrays
# model.to_json() -> Only the architecture
# model.to_yaml() -> Only the architecture also
# model.save_weights('autoencoder1_weights.h5')
# model.load_weights('autoencoder1_weights.h5', by_name=False) ->
  # load by name if using different architecture

# model.layers # List of layers addedd to the model
#