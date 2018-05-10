#FEEDFORWARD NEURAL NETWORK(FNN) WITH BACK PROPAGATION SOLUTION FOR XOR
#Build with Tensorflow
#Copyright 2018 Denis Rothman MIT License. See LICENSE.
import random
import tensorflow as tf

#I.data flow graph

x_ = tf.placeholder(tf.float32, shape=[4,2], name = 'x-input-predicates')
y_ = tf.placeholder(tf.float32, shape=[4,1], name = 'y-expected-output')

W1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name = "Weights1")
W2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name = "Weights2")

B1 = tf.Variable(tf.zeros([2]), name = "Bias1")
B2 = tf.Variable(tf.zeros([1]), name = "Bias2")

LS = tf.sigmoid(tf.matmul(x_, W1) + B1)

Output = tf.sigmoid(tf.matmul(LS, W2) + B2)

#cost = tf.reduce_mean(( (y_ * tf.log(Output)) + ((1 - y_) * tf.log(1.0 - Output)) ) * -1)
cost = tf.reduce_mean(tf.square(y_-Output))
train_step = tf.train.GradientDescentOptimizer(0.10).minimize(cost)
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


#II.data

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

#III.data flow graph computation
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(20001):
	sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
	if epoch % 10000 == 0:
		print('Epoch ', epoch)
		print('Output ', sess.run(Output, feed_dict={x_: XOR_X, y_: XOR_Y}))
		print('Weights 1 ', sess.run(W1))
		print('Bias 1 ', sess.run(B1))
		print('Weights 2 ', sess.run(W2))
		print('Bias 2 ', sess.run(B2))
		print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
                
