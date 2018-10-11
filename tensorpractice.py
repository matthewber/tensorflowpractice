#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import skimage

from keras.models import Sequential
from keras.layers import Dense, Activation

def load_data(training_directory):
    directories = [d for d in os.listdir(training_directory) if os.path.isdir(os.path.join(training_directory, d))]
    labels = []
    images = []
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
print(images.ndim)
print(images.size)

images[0]
