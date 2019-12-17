import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from timer import timer

from load_basic import DATA_FOLDER, EXPER_FOLDER, DATA_SIZE
from read_data import ideal_data_generator, exper_data_generator, DATA_FEATURES, EXPER_FEATURES

learning_rate  = 1e-3
batch_size     = 128
training_steps = 4000
epoch          = 50
display_step   = 10

num_features =  DATA_FEATURES # number of wave length separate.
print("data features: ", num_features)

num_hidden_1 = 1313 # 1st layer num features.
num_hidden_2 = 256 # 2nd layer num features (the latent dim).
num_hidden_3 = 16
num_hidden_4 = 1

train_size = int(0.8*DATA_SIZE)
#  test_size = int(0.3*DATA_SIZE)

dataset = tf.data.Dataset.from_generator(
    ideal_data_generator,
    (tf.float32, tf.float32, tf.float32),
    (tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None))
)

dataset   = dataset.shuffle(6000).repeat(epoch).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
train_set = dataset.take(train_size)
test_set  = dataset.skip(train_size)

random_normal = tf.initializers.RandomNormal()

weights = {
    'h1': tf.Variable(random_normal([num_features, num_hidden_1])),
    'h2': tf.Variable(random_normal([num_hidden_1, num_hidden_2])),
    'h3': tf.Variable(random_normal([num_hidden_2, num_hidden_3])),
    'h4': tf.Variable(random_normal([num_hidden_3, num_hidden_4])),
}
biases = {
    'b1': tf.Variable(random_normal([num_hidden_1])),
    'b2': tf.Variable(random_normal([num_hidden_2])),
    'b3': tf.Variable(random_normal([num_hidden_3])),
    'b4': tf.Variable(random_normal([num_hidden_4])),
}



# Building the encoder.
def ANN(x):
    # Encoder Hidden layer with sigmoid activation.
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Encoder Hidden layer with sigmoid activation.
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    layer_out = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))
    return layer_out



# Mean square loss between original images and reconstructed ones.
def lossfunc(predicted, original):
    return tf.reduce_mean(tf.pow(original-predicted, 2))

# Adam optimizer.
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)


# Optimization process. 
def run_optimization(vtemp, rtemp, x):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        predicted = ANN(x)
        loss = lossfunc(predicted, vtemp)

    # Variables to update, i.e. trainable variables.
    trainable_variables = list(weights.values()) + list(biases.values())

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss


# Run training for the given number of steps.
residual = []
@timer
def run():
    for step, (v, r, x) in enumerate(train_set.take(training_steps)):
        #  try:
        #      if step == 0 or step == 1:
        #          continue
        #      print(step)
        #      print(len(x))
        #      for i, item in enumerate(x[:2]):
        #          plt.title(v[i])
        #          plt.plot(item)
        #          plt.show()
        #      time.sleep(1)
        #  except:
        #      break
        loss = run_optimization(v, r, x)
        residual.append(loss)
        if step % display_step == 0:
            print("step: %i, loss: %f" % (step, loss))
run()

plt.plot(residual)
plt.show()

#
#  # Encode and decode images from test set and visualize their reconstruction.
#  n = 1
#  predict = []
#  real = []
#  print("here")
#  for step, (v, r, x) in enumerate(test_set.take(n)):
#      print(step)
#      # Encode and decode the digit image.
#      predicted = ANN(x)
#      predict += list(predicted)
#      real += list(v)
#
#  plt.plot(predict)
#  plt.plot(real)
#  plt.show()


