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
        self.sample_length = input_length
        self.latent_length = latent_length
        self.num_classes = num_classes
        print ("Initializing autoencoder network...")

    def autoencoder (self):
        x_in = Input(shape = (self.sample_length), name = 'input')

        x = layers.Dense(128)(x_in)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(64)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(self.latent_length)(x)
        x = layers.LeakyReLU()(x)
        bottleneck = x
        x = layers.Dense(64)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(128)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(200)(x)
        y = x
        out = [bottleneck, y]
        return Model(x_in, out, name = "oxynet")
