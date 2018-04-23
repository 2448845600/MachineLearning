import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;

a1 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a1')
a2 = tf.Variable(tf.constant(1), name='a2')
a3 = tf.Variable(tf.ones(shape=[2, 3]), name='a3')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(a1))
    print(sess.run(a2))
    print(sess.run(a3))