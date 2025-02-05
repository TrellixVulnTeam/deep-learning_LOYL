'''
    Morgan Ciliv
    HRI Lab
    April 2017

    Partial Differential Equations Example from tensorflow.org
    
    This program is for simulating the behavior of a PDE.
        -Particularly, this program simulates the surface of a square pond
         where a few rain drops have landed on it.
         
    Main goal: Understand the structure of this example
'''


#-------------------------------------------------------------------------------
#                       Basic Setup
#-------------------------------------------------------------------------------

# Import libraries for simulation
import tensorflow as tf
import numpy as np

# Imports for visualization
import PIL.Image # instead of PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display

# Function for displaying the state of the pond's surface as an image.
def DisplayArray(a, fmt='jpeg', rng=[0, 1]):
    """Display an array as a picture."""
    a = (a -rng[0]) / float(rng[1] - rng[0])*255
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    clear_output(wait = True)
    display(Image(data=f.getvalue()))


# Begin an interactive TensorFlow session for convenience in playing around.
# Note: Regular session would also work for executable files such as this
sess = tf.InteractiveSession()

#-------------------------------------------------------------------------------
#                       Computational Convenience Functions
#-------------------------------------------------------------------------------

def make_kernel(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return tf.constant(a, dtype=1)

def simple_conv(x, k):
    """A simplified 2D convolution operation"""
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y[0, :, :, 0]

def laplace(x):
    """Compute the 2D laplacian of an array"""
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                                 [1.0, -6., 1.0],
                                 [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)

#-------------------------------------------------------------------------------
#                       Define the PDE
#-------------------------------------------------------------------------------

# Set the dimension for the square pond
N = 500

# Create pond
# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)

# Some rain drops hit a pond at random points
for n in range(40):
    a,b = np.random.randint(0, N, 2)
    u_init[a,b] = np.random.uniform()

DisplayArray(u_init, rng=[-0.1, 0.1])

# Specify details of the PDE:

# Parameters:
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# Discretized PDE update rules
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

# Operation to update the state
step = tf.group(
    U.assign(U_),
    Ut.assign(Ut_))

#-------------------------------------------------------------------------------
#                       Run The Simulation
#-------------------------------------------------------------------------------

# Initialize state to initial conditions
tf.initialize_all_variables().run() # Switched from global_variables_initializer
                                    # to initialize_all_variables() and it ran
                                    # at least

# Run 1000 steps of PDE
for i in range(1000):
    # Step simulation
    step.run({eps: 0.03, damping: 0.04})
    DisplayArray(U.eval(), rng=[-0.1, 0.1])
