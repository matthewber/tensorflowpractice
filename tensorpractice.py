#!/usr/bin/env python



import tensorflow as tf
import numpy as np
import os
import skimage

from keras.models import Sequential
from keras.layers import Dense, Activation

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs = input_layer, filters = 32, kernel_size = [5, 5], activation = tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2)
    conv2 = tf.layers.conv2d(inputs = pool2d, filters = 64, kernel_size = [5, 5], activation = tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)
    pool2_flattened = tf.reshape(-1, 7 * 7 * 64)
    dense = tf.layers.dense(inputs = pool2_flattened, units = 1024, activation = tf.nn.relu)
    dropout = tf.layers.dropout(inputs = dense, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs = dropout, units = 10)


def load_data(training_directory):
    directories = [d for d in os.listdir(training_directory) if os.path.isdir(os.path.join(training_directory, d))]
    labels = []
    image = []
    for d in directories:
        label_directory = os.path.join(training_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

root_path = "/Users/matthewberntsen/schoolwork/git/tensorflowpractice"
training_directory = os.path.join(root_path, "Training")
testing_directory = os.path.join(root_path, "Testing")

images, labels = load_data(training_directory)
images = np.array(images)
