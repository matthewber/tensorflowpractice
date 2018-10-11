#!/usr/bin/env python

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(32, input_shape=(16,)))
#above line adds first layer with input arrays of (*,16)
#and outputs an array of shape (*, 32)
model.add(Dense(32))

x1 = tf.constant([1, 2, 4, 5])
x2 = tf.constant([2, 2, 1, 4])
result = tf.multiply(x1, x2)
with tf.Session() as sesh:
    output = sesh.run(result)
    print(output)
