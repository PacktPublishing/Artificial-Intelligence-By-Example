""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Initial Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

LOGDIR = 'rlog/RNN/'


# Import MNIST data

# once imported, comment
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("rlog/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Hyperparameters
learning_rate = 0.001
training_steps = 200   #training_steps = 10000 but during dev 200
batch_size = 128
display_step = 200

# Network hyperparameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
with tf.name_scope('input'):
    X = tf.placeholder("float", [None, timesteps, num_input],name='x-input')
    Y = tf.placeholder("float", [None, num_classes],name='y-input')

def variable_summaries(var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)       #the mean of the elements of a tensor
      tf.summary.scalar('mean', mean)  
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))#var-mean will be displayed
      tf.summary.scalar('stddev', stddev) # calcultes the standard deviation of the input 
      tf.summary.scalar('max', tf.reduce_max(var)) #calculates the maximum of the elements in the tensor
      tf.summary.scalar('min', tf.reduce_min(var)) #calculates the minimum of the elements in the tensor
      tf.summary.histogram('histogram', var)       # summarizes var for the histogram tab on Tensorboard



# Define weights
with tf.name_scope('weights'):
    weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([num_classes]))}



def RNN(x, weights, biases):
        
    
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    with tf.name_scope('RNN'):
        x = tf.unstack(x, timesteps, 1)
        variable_summaries(x)
        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        #variable_summaries(lstm_cell)
        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        #variable_summaries(outputs)
        #variable_summaries(states)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


with tf.name_scope('Logits_Softmax'):
    logits = RNN(X, weights, biases)
with tf.name_scope('Prediction'):
    prediction = tf.nn.softmax(logits)

# Define loss and optimizer
with tf.name_scope('Loss_Train'):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)
    #train_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
    #train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)
    #test_writer = tf.summary.FileWriter(LOGDIR + '/test')
    tf.global_variables_initializer().run()



    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
