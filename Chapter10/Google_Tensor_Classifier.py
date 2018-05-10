# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" This TensforFlow source code has not been changed.
However comments have been added
by Denis Rothman for "Artificial Intelligence by Example" published by PackPub.
This program is a MINST classifier.
It uses tf.name.scote to make a dataflow graph structure that can be displayed in
Tensorboard. Tensorboard_reader.py reader made avaiable by Denis Rothman on GitHub
can be used to display the summaries in Tensorboard. Several Tensorboard dashboards
will contain information written by this program. The naming summary tags were written
so that the information is grouped meaningfully in TensorBoard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#argparse is a Python command-line parsing module
import argparse
import os
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#FLAGS are command line runtime parameters, directories, for example or other information
# such as the learning rate
#FLAGS starts 0out with a None property
FLAGS = None

def train():
  # Importing MNIST data:
  #fake_data:  This is unit testing data
  #one_hot=true: To compare data for training and evaluation, a loss function will measure VALUES
  #              In this model, cross-entropy is applied.
  #              To use croos-entropy, the labels must be first converted to one-hot encoding:
  # [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  # ...]
  # Now that the labels are encoded in values, cross-entropy can be applied.
  mnist = input_data.read_data_sets(FLAGS.data_dir,one_hot=True,fake_data=FLAGS.fake_data)


  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders operations. They define the input shape.
  # tf.name_scope name is unique and is a prefix to each item
  # created in that scoce. 'x-input', for example will be displayed
  # under 'input' in Tensorboard. Then when you click on input, you
  # will see x-input and y-input.
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  # The image is reshaped
  #tf.summmary.image provides an interesting summmary(information sent to Tensorboard)to
  #actually visualize and image after the input and reshape function.
  # This can be visualized on the Tensorboard "IMAGES" dashboard. 
  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  #Weights are mostly initialized the tf.truncated normal tensor.
  # Truncated normal distribution is a normal(or Laplace-Gauss) distribution of random variables.
  # The weights are distributed randomly which avoids initializes 0 weights that will
  # block the network's learning progression when using loss functions.
  # stddev will allow a 0.1 devisionofthe normal distribution and then truncate it.
  # A "seed" option is available which can be used to create a random seed for the distribution.
  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  # Populates a bias constant tensor with a value (here 0.1) or list of values.
  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  #Defining many summaries for Tensorboard Visualization of:
  # a)weights that will send weights to this function :variable_summaries(weights)
  # b)biases that will send biasesto this function: variable_summaries(biases)
  # Tensorboard will then display charts of summaries described below on the related
  # tensorboard dashboards
  # These weight and bias summaries provide a clear view, in Tensorboard of the direction
  # the values are taking during training.It will help fine-tune the hyperparameters when
  # things go wrong or training does not go fast enough.
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

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Interesting reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  hidden1 = nn_layer(x, 784, 500, 'layer1')

  
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    # Cross entropy is calculated using probablity error using a softmax function with
    # logits which are the unscaled log probablities.
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries
  # "feed_dict" will send data to the Tensorflow dataflow graph structere defined above.
  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


#Argparse is the argument parser
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

