'''
    Morgan Ciliv
    HRI Lab
    April 2017
    
    Deep MNIST for Experts tutorial on TensorFlow.org
    
    Multilayer Convolutional Network (2nd part of the tutorial)
        - Previously we got ~92 % accuracy
        - The goal of this better method is to get 99.2 %
        
    Note: Running this code can take a while, up to 30 min depending on
    the processor.
    
    Obtained accuracy of 0.9926 when run.
'''

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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

#                       Convolutional Neural Network
#-------------------------------------------------------------------------------
#                       Weight Initialization
#
# Need to create a lot of weights and biases
# Generalize weights with a small amount of noise symmetry breaking to prevent
# 0 gradients
#
# Also initialize them to slightly positive vals to avoid "dead neurons"
#-------------------------------------------------------------------------------

# Define 2 functions for modularity of these ideas
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#-------------------------------------------------------------------------------
#                       Convolution and Pooling
#
# Handle the boundaries, set stride size
#-------------------------------------------------------------------------------

# Stride of 1, pooling is over 2x2 blocks
# Abstract the operations into functions:
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

#-------------------------------------------------------------------------------
#                       First Convolutional Layer
#
# Steps: Convolution then max pooling
#-------------------------------------------------------------------------------

# Convolution will compute 32 features for each 5x5 patch
# First 2 dimensions are the patch size, then # of input then # of output
# channels
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Apply layer by first reshape x to a 4d tensor
# 2nd dimension and 3rd dimension: image width then height
# Final dimension: number of channels
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolve x_image with the weight tensor add bias, apply the ReLU (Rectifier
# Linear Unit) function, and max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#-------------------------------------------------------------------------------
#                       Second Convoluational Layer
#
# Building a deep network entails stacking several layers of this type.
#-------------------------------------------------------------------------------

# This second layer has 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#-------------------------------------------------------------------------------
#                       Densely Connected Layer
#-------------------------------------------------------------------------------

# The size has been reduced to 7x7.
# Add a fully connected layer with 1024 neurons to allow processing on the
# entire image
# Reshape the tensor from the pooling layer into a batch of vectors, multiply
# by a weight matrix, add a bias, apply ReLU
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#-------------------------------------------------------------------------------
#                       Dropout
#
# To reduce overfitting, we apply dropout before the readout layer
# Create a placeholder for the probability that a neuron's output is kept during
# dropout allowing us to turn on dropout during training and off during testing
#-------------------------------------------------------------------------------

# tf.nn.dropout op automatically handles scaling neuron outputs in addition to
# masking them. Therefore, dropout works without additional scaling
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#-------------------------------------------------------------------------------
#                       Readout Layer
#
# Add a layer just like the one layer for the SoftMax regression
#-------------------------------------------------------------------------------

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#-------------------------------------------------------------------------------
#                       Train and Evaluate the Model
#
# Nearly the same as the SoftMax network, but with the following improvements:
# - Use sophisticated ADAM optimizer instead of gradient descent
# - Additional parameter keep_prob in the feed_dict to control the dropout rate
# - Add logging to every 100th iteration in the training process
# Note: Performance is nearly identical w/ and w/out dropout. Dropout is very
# effective at reducing overfitting, but is most useful for large networks.
#-------------------------------------------------------------------------------

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
