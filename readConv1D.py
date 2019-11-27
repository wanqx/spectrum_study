from __future__ import absolute_import, division, print_function
from conv1D import ConvNet
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import matplotlib.pyplot as plt

from read_data import ideal_data_generator, exper_data_generator, DATA_FEATURES, EXPER_FEATURES

new_model = ConvNet()
new_model.load_weights('testSave')
#  pred = new_model(x_test)

dataset = tf.data.Dataset.from_generator(
    ideal_data_generator,
    (tf.float32, tf.float32, tf.float32),
    (tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None)))

for item in dataset.shuffle(100).take(2):
    vtemp, rtemp, data = item
    pred = new_model(data)
    print('t real: ', vtemp, 'pred: ', pred)
