#FEEDFORWARD NEURAL NETWORK(FNN) WITH BACK PROPAGATION 
#Build with Tensorflow
#Copyright 2018 Denis Rothman MIT License. See LICENSE.
import tensorflow as tf
import os
PATH = os.getcwd()
LOG_DIR = PATH+ '/LOGS/'

#I.data flow graph

with tf.name_scope("input"):
        x_ = tf.placeholder(tf.float32, shape=[4,2], name = 'x-input-predicates')#placeholder is an operation supplied by the feed
        tf.summary.image('input store products', x_, 10)

with tf.name_scope("input_expected_ouput"):      
        y_ = tf.placeholder(tf.float32, shape=[4,1], name = 'y-expected-output')
        tf.summary.image('input expected classification', x_, 10)

with tf.name_scope("Weights1"):
        W1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name = "Weights1")
        tf.summary.image("Weights1", W1, 10)

with tf.name_scope("Weights2"):        
        W2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name = "Weights2")
        tf.summary.image("Weights2", W2, 10)


with tf.name_scope("Bias1"):        
        B1 = tf.Variable(tf.zeros([2]), name = "Bias1")
        tf.summary.image("Bias1", B1, 10)

with tf.name_scope("Bias2"):        
        B2 = tf.Variable(tf.zeros([1]), name = "Bias2")
        tf.summary.image("Bias2", B2, 10)

with tf.name_scope("Hidden_Layer__Logistic_Sigmoid"):        
        LS = tf.sigmoid(tf.matmul(x_, W1) + B1)
        tf.summary.image("Hidden_Layer__Logistic_Sigmoid", LS, 10)

with tf.name_scope("Ouput__Layer_Logistic_Sigmoid"):        
        Output = tf.sigmoid(tf.matmul(LS, W2) + B2)
        tf.summary.image("Output_Layer_Logistic_Sigmoid", Output, 10)

with tf.name_scope("Error_Loss"):        
        cost = tf.reduce_mean(( (y_ * tf.log(Output)) + ((1 - y_) * tf.log(1.0 - Output)) ) * -1)
        tf.summary.image("Error_Loss", cost, 10)
        
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


#II.data flow architecture graph writer
init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
sess.run(init)

#feeding the data and running graph computation is called here.
#insert your lines here

writer.close()
