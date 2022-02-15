'''
Sunwoo Lee Ph.D.
Postdoctoral Researcher
University of Southern California
<sunwool@usc.edu>
'''

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.regularizers import l2

class Network ():
    def __init__ (self, input_length, latent_length, num_classes):
        self.weight_decay = 1e-4
        self.batch_norm_momentum = 0.99
        self.batch_norm_epsilon = 1e-5
        self.sample_length = input_length
        self.latent_length = latent_length
        self.num_classes = num_classes
        self.filter_size = 256
        self.kernel_size = 4
        self.num_layers = 10
        self.initializer = tf.keras.initializers.GlorotNormal(seed = int(time.time()))
        print ("Initializing a simple feed-forward network...")

    def autoencoder (self):
        self.regularizer = l2(self.weight_decay)
        x_in = Input(shape = (self.sample_length, 1), name = 'input')
        filter_size = self.filter_size

        x = layers.Conv1D(filters = self.filter_size,
                          kernel_size = 3,
                          strides = 1,
                          padding = "same")(x_in)
        x = layers.ReLU()(x)

        x = layers.Conv1D(filters = self.filter_size * 2,
                          kernel_size = 3,
                          strides = 2,
                          padding = "same")(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(filters = self.filter_size * 2,
                          kernel_size = 3,
                          strides = 2,
                          padding = "same")(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(filters = self.filter_size * 2,
                          kernel_size = 3,
                          strides = 2,
                          padding = "same")(x)
        x = layers.ReLU()(x)

        hidden = layers.Flatten()(x)
        bottleneck = layers.Dense(self.latent_length)(hidden)

        filter_length = int(self.sample_length / 8)
        size = filter_length * self.filter_size * 2
        x = layers.Dense(size)(bottleneck)
        x = tf.reshape(x, [-1, filter_length, self.filter_size * 2])
        x = layers.Conv1DTranspose(filters = self.filter_size * 2,
                                   kernel_size = 3,
                                   strides = 1,
                                   padding = "same")(x)
        x = layers.ReLU()(x)

        x = layers.Conv1DTranspose(filters = self.filter_size * 2,
                                   kernel_size = 3,
                                   strides = 2,
                                   padding = "same")(x)
        x = layers.ReLU()(x)

        x = layers.Conv1DTranspose(filters = self.filter_size * 2,
                                   kernel_size = 3,
                                   strides = 2,
                                   padding = "same")(x)
        x = layers.ReLU()(x)

        x = layers.Conv1DTranspose(filters = self.filter_size,
                                   kernel_size = 3,
                                   strides = 2,
                                   padding = "same")(x)
        x = layers.ReLU()(x)

        x = layers.Conv1DTranspose(filters = 1,
                                   kernel_size = 4,
                                   strides = 1,
                                   padding = "same")(x)
        y = x
        out = [bottleneck, y]
        return Model(x_in, out, name = "oxynet")

    def build_model (self):
        self.regularizer = l2(self.weight_decay)
        x_in = Input(shape = (self.sample_length), name = 'input')
        x = layers.Dense(self.filter_size, 
                         kernel_regularizer = self.regularizer)(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        #x = activations.tanh (x)
        size = self.filter_size
        for i in range (self.num_layers):
            #if i % 2 == 0:
            #    shortcut = x
            x = layers.Dense(size, 
                             kernel_regularizer = self.regularizer)(x)
            x = layers.BatchNormalization()(x)
            #if (i + 1) % 2 == 0:
            #    x = x + shortcut
            x = layers.LeakyReLU() (x)
            #x = activations.tanh (x)
        y = layers.Dense(14, 
                         kernel_regularizer = self.regularizer,
                         activation = 'softmax')(x)
        return Model(x_in, y, name = "oxynet")

    def build_model2 (self):
        self.regularizer = l2(self.weight_decay)
        x_in = Input(shape = (self.sample_length, 1), name = 'input')
        filter_size = self.filter_size

        x = layers.Conv1D(filters = self.filter_size,
                          kernel_size = self.kernel_size,
                          strides = 1,
                          kernel_regularizer = self.regularizer,
                          padding = "same")(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        #x = activations.tanh(x)

        for i in range (self.num_layers):
            x = layers.Conv1D(filters = self.filter_size,
                              kernel_size = self.kernel_size,
                              strides = 1,
                              kernel_regularizer = self.regularizer,
                              padding = "same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            #x = activations.tanh(x)
        x = layers.GlobalAveragePooling1D()(x)
        #x = layers.Flatten()(x)

        y = layers.Dense(5, 
                         kernel_regularizer = self.regularizer,
                         bias_regularizer = self.regularizer,
                         activation = 'softmax')(x)
        return Model(x_in, y, name = "oxynet")
