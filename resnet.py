from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from timer import timer
from load_basic import DATA_FOLDER, EXPER_FOLDER, DATA_SIZE
from read_data import ideal_data_generator, exper_data_generator, DATA_FEATURES, EXPER_FEATURES

class ResNet50():
    _L2_WEIGHT_DECAY = 1e-4

    @staticmethod
    def _gen_l2_regularizer(use_l2_regularizer=True):
        return tf.keras.regularizers.l2(ResNet50._L2_WEIGHT_DECAY) if use_l2_regularizer else None

    @staticmethod
    def _identity_block(input, filters, use_l2_regularizer):
        filter1, filter2, filter3 = filters

        x = tf.keras.layers.Conv1D(
            filters=filter1,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(input)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv1D(
            filters=filter2,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv1D(
            filters=filter3,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    @staticmethod
    def _conv_block(input, filters, stride, use_l2_regularizer):
        filter1, filter2, filter3 = filters

        x = tf.keras.layers.Conv1D(
            filters=filter1,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(input)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv1D(
            filters=filter2,
            kernel_size=3,
            strides=stride,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv1D(
            filters=filter3,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        shortcut = tf.keras.layers.Conv1D(
            filters=filter3,
            kernel_size=1,
            strides=stride,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(input)
        shortcut = tf.keras.layers.BatchNormalization(axis=1)(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def __init__(self, global_batch_size, img_size, learning_rate=3e-4, use_l2_regularizer=True):

        self.img_size = img_size
        self.learning_rate = learning_rate
        self.global_batch_size = global_batch_size
        self.use_l2_regularizer = use_l2_regularizer

        # image is HWC (normally e.g. RGB image) however data needs to be NCHW for network
        # self.inputs = tf.keras.Input(shape=(img_size[2], img_size[0], img_size[1]))
        self.model = self._build_model()

        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_model(self):
        # reinterpreted from: https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet_model.py


        class RES50(tf.keras.Model):
            def __init__(self, cls):
                super().__init__()
                self.f1 = tf.keras.layers.Conv1D(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    activation=None,
                    use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=ResNet50._gen_l2_regularizer(cls.use_l2_regularizer),
                    data_format='channels_first')
                self.f2 = tf.keras.layers.BatchNormalization(
                    axis=1)
                self.f3 = tf.keras.layers.Activation('relu')

                self.f4 = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')
                self.cls = cls

            def call(self, x, training):
                cls = self.cls
                x = tf.reshape(x, [-1, DATA_FEATURES, 1])
                x = self.f1(x)
                x = self.f2(x)
                x = self.f3(x)
                x = self.f4(x)
                x = ResNet50._conv_block(x, [64, 64, 256], stride=1, use_l2_regularizer=cls.use_l2_regularizer)
                x = ResNet50._identity_block(x, [64, 64, 256],
                                         use_l2_regularizer=cls.use_l2_regularizer)
                x = ResNet50._identity_block(x, [64, 64, 256],
                                         use_l2_regularizer=cls.use_l2_regularizer)

                x = ResNet50._conv_block(x, [128, 128, 512], stride=2,
                                         use_l2_regularizer=cls.use_l2_regularizer)
                x = ResNet50._identity_block(x, [128, 128, 512],
                                             use_l2_regularizer=cls.use_l2_regularizer)
                x = ResNet50._identity_block(x, [128, 128, 512],
                                             use_l2_regularizer=cls.use_l2_regularizer)
                x = ResNet50._identity_block(x, [128, 128, 512],
                                             use_l2_regularizer=cls.use_l2_regularizer)

                x = ResNet50._conv_block(x, [256, 256, 1024], stride=2,
                                         use_l2_regularizer=cls.use_l2_regularizer)
                x = ResNet50._identity_block(x, [256, 256, 1024],
                                             use_l2_regularizer=cls.use_l2_regularizer)
                x = ResNet50._identity_block(x, [256, 256, 1024],
                                             use_l2_regularizer=cls.use_l2_regularizer)
                x = ResNet50._identity_block(x, [256, 256, 1024],
                                             use_l2_regularizer=cls.use_l2_regularizer)
                x = ResNet50._identity_block(x, [256, 256, 1024],
                                             use_l2_regularizer=cls.use_l2_regularizer)
                x = ResNet50._identity_block(x, [256, 256, 1024],
                                             use_l2_regularizer=cls.use_l2_regularizer)

                x = ResNet50._conv_block(x, [512, 512, 2048], stride=2,
                                         use_l2_regularizer=cls.use_l2_regularizer)
                x = ResNet50._identity_block(x, [512, 512, 2048],
                                             use_l2_regularizer=cls.use_l2_regularizer)
                x = ResNet50._identity_block(x, [512, 512, 2048],
                                             use_l2_regularizer=cls.use_l2_regularizer)

                # output_layer_name5 is tensor with shape <batch_size>, 2048, <img_size>/32, <img_size>/32
                # downsample_factor = 32
                rm_axes = [2, 2]
                x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, rm_axes), name='reduce_mean')(x)

                logits = tf.keras.layers.Dense(
                    1,
                    kernel_initializer='he_normal',
                    kernel_regularizer=ResNet50._gen_l2_regularizer(cls.use_l2_regularizer),
                    bias_regularizer=ResNet50._gen_l2_regularizer(cls.use_l2_regularizer),
                    activation=None,
                    name='logits')(x)
                return logits

        return RES50(self)

    def get_keras_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def set_learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.optimizer.learning_rate

    def train_step(self, inputs):
        (images, labels, loss_metric) = inputs
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)

            loss_value = self.loss_fn(labels, logits) # [Nx1]
            # average across the batch (N) with the appropriate global batch size
            loss_value = tf.reduce_sum(loss_value, axis=0) / self.global_batch_size

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, self.model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        loss_metric.update_state(loss_value)

        return loss_value

    @tf.function
    def dist_train_step(self, dist_strategy, inputs):
        per_gpu_loss = dist_strategy.experimental_run_v2(self.train_step, args=(inputs,))
        loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)

        return loss_value

    def test_step(self, inputs):
        (images, labels, loss_metric) = inputs
        logits = self.model(images, training=False)

        loss_value = self.loss_fn(labels, logits)
        # average across the batch (N) with the approprite global batch size
        loss_value = tf.reduce_sum(loss_value, axis=0) / self.global_batch_size

        loss_metric.update_state(loss_value)

        return loss_value

    @tf.function
    def dist_test_step(self, dist_strategy, inputs):
        per_gpu_loss = dist_strategy.experimental_run_v2(self.test_step, args=(inputs,))
        loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)
        return loss_value

if __name__ == "__main__":
    learning_rate  = 1e-4
    batch_size     = 128
    training_steps = 1000
    epoch          = 20
    display_step   = 1

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

    resnet = ResNet50(batch_size, DATA_FEATURES, learning_rate)
    LOSS = []
    loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    # Run training for the given number of steps.
    for step, (v, r, batch_x) in enumerate(train_set.take(training_steps), 1):
        # Run the optimization to update W and b values.
        inputs = (batch_x, v, loss_metric)
        if step % display_step == 0:
            resnet.train_step(inputs)
            LOSS.append(loss_metric.result())
            print('step {}: Loss {}'.format(step, loss_metric.result()))

    plt.plot(LOSS)
    plt.show()
    plt.plot(np.log(LOSS))
    plt.show()
    # Test model on validation set.
    #  pred = resnet(x_test)
    ResNet50.save_weights("testResNet")
