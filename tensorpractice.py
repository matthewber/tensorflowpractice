#!/usr/bin/env python

import tensorflow as tf

x1 = tf.constant([1, 2, 4, 5])
x2 = tf.constant([2, 2, 1, 4])
result = tf.multiply(x1, x2)
se = tf.Session()
print(se.run(result))
se.close()
