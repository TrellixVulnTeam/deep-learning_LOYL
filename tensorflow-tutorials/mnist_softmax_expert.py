'''
    Morgan Ciliv
    HRI Lab
    April 2017
    
    This is the softmax model part of the "MNIST For ML experts" tutorial
    on TensorFlow.org.
    
    Obtained accuracy of 0.9177 when run.
'''

#                       Setup
#-------------------------------------------------------------------------------
#                       Load MNIST Data
#-------------------------------------------------------------------------------

# Get the MNIST data, splitting into training and testing set automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#-------------------------------------------------------------------------------
#                       Start TensorFlow InteractiveSession
#-------------------------------------------------------------------------------

# Import TensorFlow
import tensorflow as tf

# Allows you to interleave operations which build a computation graph with ones
# that run the graph
sess = tf.InteractiveSession()

#                       Build a Softmax Regression Model
#-------------------------------------------------------------------------------
#                       Placeholders
#-------------------------------------------------------------------------------

# A 784 by 1 matrix that will hold values
x = tf.placeholder(tf.float32, [None, 784])

# Will hold the 10 labels for the 10 numbers, 0 - 9 for the cross-entropy func
y_ = tf.placeholder(tf.float32, [None, 10])

#-------------------------------------------------------------------------------
#                       Variables
#-------------------------------------------------------------------------------

# Initialize the Weight matrix and bias vector with 0s
W = tf.Variable(tf.zeros([784, 10])) # bec. 784 input features and 10 outputs
b = tf.Variable(tf.zeros([10])) # bec. 10 classes

# Variables must be initialized using that session in order to use it
sess.run(tf.initialize_all_variables())

#-------------------------------------------------------------------------------
#                       Predicted Class and Loss Function
#-------------------------------------------------------------------------------

# Regression Model
y = tf.matmul(x, W) + b

# Loss indicates how bad the model's prediction
# Goal is to minimize that while training across all examples
#
# This loss func is the cross-entropy between the target and the softmax
# activation function applied to the model's prediction w/ stable formulation
#
# Note: the internal func also applies the softmax function on the models
# unnormalized model prediction
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

#                       Train the Model
#
# We can do this because we have our model and training loss function
#-------------------------------------------------------------------------------

# Runs gradient descent in order to obtain optimal weights
# 0.5 is the step length for descending the cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Runs the training set 1000 times
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#-------------------------------------------------------------------------------
#                       Evaluate the Model
#-------------------------------------------------------------------------------

# See if the prediction matches the truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Gets the accuracy of the test
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
