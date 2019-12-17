from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import matplotlib.pyplot as plt
from timer import timer
from load_basic import DATA_FOLDER, EXPER_FOLDER, DATA_SIZE, NET_NAME
from read_data import ideal_data_generator, exper_data_generator, DATA_FEATURES, EXPER_FEATURES


learning_rate  = 1e-3
batch_size     = 1
epoch          = 5
training_steps = epoch*DATA_SIZE
display_step   = 1
print(training_steps)

num_features =  DATA_FEATURES # number of wave length separate.
print("data features: ", num_features)

#  num_hidden_1 = 512 # 1st layer num features.
#  num_hidden_2 = 128 # 2nd layer num features (the latent dim).
#  num_hidden_3 = 32
#  num_hidden_4 = 1

train_size = int(0.8*DATA_SIZE)
#  test_size = int(0.3*DATA_SIZE)

dataset = tf.data.Dataset.from_generator(
    ideal_data_generator,
    (tf.float32, tf.float32, tf.float32),
    (tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None))
)

train_set = dataset.take(train_size)
test_set  = dataset.skip(train_size)

train_set = train_set.shuffle(train_size).repeat(epoch).batch(batch_size).prefetch(epoch)
#  train_set = train_set.repeat(epoch).batch(batch_size).prefetch(epoch)

class ConvNet(Model):
    # Set layers.
    def __init__(self):
        super().__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = layers.Conv1D(16, kernel_size=5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = layers.Conv1D(32, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.conv3 = layers.Conv1D(32, kernel_size=2, activation=tf.nn.relu)
        self.conv4 = layers.Conv1D(64, kernel_size=2, activation=tf.nn.relu)
        self.conv5 = layers.Conv1D(64, kernel_size=2, activation=tf.nn.relu)
        self.conv6 = layers.Conv1D(64, kernel_size=2, activation=tf.nn.relu)
        self.conv7 = layers.Conv1D(64, kernel_size=2, activation=tf.nn.relu)
        self.conv8 = layers.Conv1D(64, kernel_size=2, activation=tf.nn.relu)
        self.conv8 = layers.Conv1D(64, kernel_size=2, activation=tf.nn.relu)
        self.conv9 = layers.Conv1D(64, kernel_size=2, activation=tf.nn.relu)
        self.conv10 = layers.Conv1D(128, kernel_size=2, activation=tf.nn.relu)
        self.conv11 = layers.Conv1D(128, kernel_size=2, activation=tf.nn.relu)
        #  self.conv12 = layers.Conv1D(128, kernel_size=2, activation=tf.nn.relu)
        #  self.conv13 = layers.Conv1D(128, kernel_size=2, activation=tf.nn.relu)
        #  self.conv14 = layers.Conv1D(128, kernel_size=2, activation=tf.nn.relu)
        #  self.conv15 = layers.Conv1D(128, kernel_size=2, activation=tf.nn.relu)
        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc2 = layers.Dense(2048)
        self.dropout2 = layers.Dropout(rate=0.5)
        #  self.fc1 = layers.Dense(1024)
        self.fc0 = layers.Dense(512)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout1 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(1)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = tf.cast(x, dtype=tf.float32)
        x = tf.reshape(x, [-1, DATA_FEATURES, 1])
        x = self.conv1(x)
        #  x = self.conv2(x)
        #  x = self.conv3(x)
        #  x = self.conv4(x)
        #  x = self.conv5(x)
        #  x = self.conv6(x)
        #  x = self.conv7(x)
        #  x = self.conv8(x)
        #  x = self.conv9(x)
        #  x = self.conv10(x)
        #  x = self.conv11(x)
        #  x = self.conv12(x)
        #  x = self.conv13(x)
        #  x = self.conv14(x)
        #  x = self.conv15(x)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.dropout2(x, training=is_training)
        #  x = self.fc1(x)
        x = self.fc0(x)
        x = self.dropout1(x, training=is_training)
        x = self.out(x)
        #  if not is_training:
        #      # tf cross entropy expect logits without softmax, so only
        #      # apply softmax when not training.
        #      x = tf.nn.softmax(x)
        return x

# Build neural network model.
conv_net = ConvNet()

# Cross-Entropy Loss.
# Note that this will apply 'softmax' to the logits.

def lossfunc(pred, ori):
    return tf.reduce_mean(tf.pow(ori-pred, 2))
    #  cp = tf.math.abs(tf.subtract(pred, ori))
    #  return tf.reduce_mean(tf.cast(cp, tf.float32))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.math.abs(tf.subtract(y_pred, y_true))
    return 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = conv_net(x, is_training=True)
        # Compute loss.
        loss = lossfunc(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = conv_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))

if __name__=="__main__":
    LOSS = []
    ACC = []
    # Run training for the given number of steps.
    for step, (v, r, batch_x) in enumerate(train_set.take(training_steps), 1):
        # Run the optimization to update W and b values.
        run_optimization(batch_x, v)

        if step % display_step == 0:
            pred = conv_net(batch_x)
            loss = lossfunc(pred, v)
            LOSS.append(loss)
            #  acc = accuracy(pred, v)
            #  ACC.append(acc)
            #  print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
            print("step: %i, loss: %f" % (step, loss))

    plt.plot(LOSS)
    plt.show()
    plt.plot(np.log(LOSS))
    plt.show()
    #  plt.plot(ACC)
    #  plt.show()
    # Test model on validation set.
    #  pred = conv_net(x_test)
    conv_net.save_weights(NET_NAME)
    #
    #  new_model = ConvNet()
    #  new_model.load_weights('testSave')
    #  pred = new_model(x_test)
    #  print("Test Accuracy2: %f" % accuracy(pred, y_test))

#
#  #  Visualize predictions.
#  import matplotlib.pyplot as plt
#
#
#  # Predict 5 images from validation set.
#  n_images = 5
#  test_images = x_test[:n_images]
#  predictions = conv_net(test_images)
#
#  # Display image and model prediction.
#  for i in range(n_images):
#      plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
#      plt.show()
#      print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))
#
