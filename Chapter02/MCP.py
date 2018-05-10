# McCulloch Pitt Neuron built with Tensorflow and represented with
# Tensorboard
# Copyright 2018 Denis Rothman MIT License. See LICENSE.

import tensorflow as tf
import numpy as np
import os

PATH = os.getcwd()

LOG_DIR = PATH+ '/output/'



# 1.Configuration to optimize CPU performance by defining
#   thread pools. For this example 4 is enough. A variable
#   replace the constant
config = tf.ConfigProto(
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4
)

# 2.Defining the x values, their w weights,the b bias,y weight calculation
#   and the s sigmoid activation function

x = tf.placeholder(tf.float32, shape=(1, 5), name='x')
w = tf.placeholder(tf.float32, shape=(5, 1), name='w')
b = tf.placeholder(tf.float32, shape=(1), name='b')
y = tf.matmul(x, w) + b
s = tf.nn.sigmoid(y)


# 3.Starting a session providing constants as weight inputs
#   The Perceptron,a neuron that can learn its weights, will provide  with our present day
#   automatic weight calculations
with tf.Session(config=config) as tfs:
    tfs.run(tf.global_variables_initializer())
    
    w_t = [[.1, .7, .75, .60, .20]]
    x_1 = [[10, 2, 1., 6., 2.]]
    b_1 = [1]
    w_1 = np.transpose(w_t)
    
    value = tfs.run(s, 
        feed_dict={
            x: x_1, 
            w: w_1,
            b: b_1
        }
    )
    
print ('value for threshold calculation',value)
print ('Availability of lx',1-value)
         
  
#___________Tensorboard________________________

#with tf.Session() as sess:

Writer = tf.summary.FileWriter(LOG_DIR, tfs.graph)
Writer.close()


def launchTensorBoard():
    import os
    os.system('tensorboard --logdir='+LOG_DIR)
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

tfs.close()
#Open your browser and go to http://localhost:6006
#Try the various options. It is a very useful tool.
#close the system window wh1en your finished.



    
