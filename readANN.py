from __future__ import absolute_import, division, print_function
from ANN import ConvNet
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import matplotlib.pyplot as plt
from load_basic import NET_NAME

from read_data import ideal_data_generator, exper_data_generator, DATA_FEATURES, EXPER_FEATURES

new_model = ConvNet()
new_model.load_weights(NET_NAME)
#  pred = new_model(x_test)

dataset = tf.data.Dataset.from_generator(
    ideal_data_generator,
    (tf.float32, tf.float32, tf.float32),
    (tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None)))

#  for item in dataset.shuffle(5000).take(10):
#      vtemp, rtemp, data = item
#      pred = new_model(data)
#      print('t real: ', vtemp, 'pred: ', pred)
predList = []

realList = []


#  k, a, b = 15.58288, 3155, 3121.634
#
#
#  for item in dataset.take(5714):
for i, item in enumerate(dataset.take(5714)):
    if i%10 != 0: continue
    vtemp, rtemp, data = item
    pred = new_model(data).numpy()[0][0]
    #  pred = (pred-b)*k+a # fit pred
    vtemp = vtemp.numpy()
    predList.append(pred)
    realList.append(vtemp)

    print('t real: ', vtemp, 'pred: ', pred)

#  for i, item in enumerate(dataset.take(5714)):
#      if i%10 != 0: continue
#      vtemp, rtemp, data = item
#      pred = new_model(data).numpy()[0][0]
#      vtemp = vtemp.numpy()
#      predList.append(pred)
#      realList.append(vtemp)
#      print('t real: ', vtemp, 'pred: ', pred)
#
#  def mean(x):
#      return sum(x)/len(x)
#
#  meanreal = mean(realList)
#  meanpred = mean(predList)
#  print("pred", meanpred)
#  print("real", meanreal)
#
#  K = []
#  for i in range(len(predList)):
#      K.append((realList[i]-meanreal)/(predList[i]-meanpred))
#  k = mean(K)
#  print("k", k)
#
#  predList = [(x-meanpred)*k+meanreal for x in predList.copy()]
plt.plot(predList)
plt.plot(realList)

plt.show()
