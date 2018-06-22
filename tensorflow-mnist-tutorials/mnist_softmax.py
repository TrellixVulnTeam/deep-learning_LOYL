'''
    Morgan Ciliv
    HRI Lab
    April 2017
    
    This is the MNIST For ML Beginners tutorial on TensorFlow.org
    
    Obtained accuracy of 0.9192 when run. 
'''

#-------------------------------------------------------------------------------
#                      Setup MNIST Data and TensorFlow
#-------------------------------------------------------------------------------

# Get the MNIST data, splitting into training and testing set automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Import TensorFlow
import tensorflow as tf

#-------------------------------------------------------------------------------
#                      Implement Softmax Regression
#-------------------------------------------------------------------------------

# A 784 by 1 matrix that will hold values
x = tf.placeholder(tf.float32, [None, 784])

# Initialize the Weight matrix and bias vector with 0s
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# This is the model definition!
# Calculate the Softmax of the scores in order to produce a prob dist
y = tf.nn.softmax(tf.matmul(x, W) + b)

#-------------------------------------------------------------------------------
#                       Training
#-------------------------------------------------------------------------------

# Will hold the 10 labels for the 10 numbers, 0 - 9 for the cross-entropy func
y_ = tf.placeholder(tf.float32, [None, 10])

# Calculates cross entropy which gets us the one hot labels
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Runs gradient descent in order to obtain optimal weights
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Operation to initialize all of the variables created above
init = tf.initialize_all_variables()

# Launches the model in a session
sess = tf.Session()
sess.run(init)

# Runs the training set 1000 times
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#-------------------------------------------------------------------------------
#                       Evaluating our Model
#-------------------------------------------------------------------------------

# See if the prediction matches the truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Gets the accuracy of the test
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
