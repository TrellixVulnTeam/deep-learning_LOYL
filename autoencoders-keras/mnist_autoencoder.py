

# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(x_test)

plt.figure()
for idx, cl in enumerate(np.unique(x_test)):
    plt.scatter(x=x_test_2d[x_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()



def main():
    try:
        load_model('my_model.h5')
    except:
        create_autoencoder()

    visualize_autoencoder()
    model.save('my_model.h5')

if __name__ == '__main__':
    main()

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

