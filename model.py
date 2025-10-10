'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2023/03/17
Sunwoo Lee, Ph.D.
<sunwool@inha.ac.kr>
'''
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2

def copy_model (model):
    new_model = tf.keras.models.clone_model(model)
    for i in range (len(model.trainable_variables)):
        new_model.trainable_variables[i].assign(model.trainable_variables[i])
    for i in range (len(model.non_trainable_variables)):
        new_model.non_trainable_variables[i].assign(model.non_trainable_variables[i])
    return new_model

class gru (tf.keras.Model):
  def __init__(self, vocab_size, sample_length, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
        states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
        return x, states
    else:
        return x

class lstm ():
    def __init__ (self, weight_decay, sample_length):
        self.weight_decay = weight_decay
        self.sample_length = sample_length

    def build_model (self):
        x_in = Input(shape = (self.sample_length))
        x = Embedding(input_dim = 10000, output_dim = 256, input_length = self.sample_length, mask_zero = True)(x_in)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(256, return_sequences=False, dropout=0.3))(x)
        y = Dense(1)(x)
        return Model(x_in, y, name = "lstm")

class vgg11 ():
    def __init__ (self, weight_decay):
        self.weight_decay = weight_decay
        self.regularizer = l2(self.weight_decay)

    def build_model (self):
        x_in = Input(shape = (28, 28, 1), name = "input")

        x = Conv2D(64,
                   (3, 3),
                   strides=(1, 1),
                   name='conv0',
                   padding='same',
                   use_bias=True,
                   kernel_regularizer = self.regularizer)(x_in)
        x = tf.nn.relu(x)
        #x = MaxPool2D((2, 2))(x)

        x = Conv2D(128,
                   (3, 3),
                   strides=(1, 1),
                   name='conv1',
                   padding='same',
                   use_bias=True,
                   kernel_regularizer = self.regularizer)(x)
        x = tf.nn.relu(x)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.3)(x)

        x = Conv2D(256,
                   (3, 3),
                   strides=(1, 1),
                   name='conv2',
                   padding='same',
                   use_bias=True,
                   kernel_regularizer = self.regularizer)(x)
        x = Conv2D(256,
                   (3, 3),
                   strides=(1, 1),
                   name='conv3',
                   padding='same',
                   use_bias=True,
                   kernel_regularizer = self.regularizer)(x)
        x = tf.nn.relu(x)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.3)(x)

        x = Conv2D(512,
                   (3, 3),
                   strides=(1, 1),
                   name='conv4',
                   padding='same',
                   use_bias=True,
                   kernel_regularizer = self.regularizer)(x)
        x = Conv2D(512,
                   (3, 3),
                   strides=(1, 1),
                   name='conv5',
                   padding='same',
                   use_bias=True,
                   kernel_regularizer = self.regularizer)(x)
        x = tf.nn.relu(x)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.3)(x)

        x = Conv2D(512,
                   (3, 3),
                   strides=(1, 1),
                   name='conv6',
                   padding='same',
                   use_bias=True,
                   kernel_regularizer = self.regularizer)(x)
        x = Conv2D(512,
                   (3, 3),
                   strides=(1, 1),
                   name='conv7',
                   padding='same',
                   use_bias=True,
                   kernel_regularizer = self.regularizer)(x)
        x = tf.nn.relu(x)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.3)(x)

        x = Flatten()(x)
        x = Dense (4096, activation='relu', kernel_regularizer = self.regularizer) (x)
        x = Dropout(0.3)(x)
        x = Dense (4096, activation='relu', kernel_regularizer = self.regularizer) (x)
        x = Dropout(0.3)(x)
        y = Dense(10, activation = 'softmax', name='fully_connected', kernel_regularizer = self.regularizer)(x)
        return Model(x_in, y, name = "vgg11")

class resnet20 ():
    def __init__ (self, weight_decay):
        self.weight_decay = weight_decay
        self.regularizer = l2(self.weight_decay)
        self.initializer = tf.keras.initializers.GlorotUniform(seed = int(time.time()))

    def res_block (self, input_tensor, num_filters, strides = (1, 1), projection = False):
        x = Conv2D(num_filters,
                   (3, 3),
                   strides = strides,
                   padding = "same",
                   use_bias = False,
                   kernel_initializer = self.initializer,
                   kernel_regularizer = self.regularizer)(input_tensor)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = Conv2D(num_filters,
                   (3, 3),
                   padding = "same",
                   use_bias = False,
                   kernel_initializer = self.initializer,
                   kernel_regularizer = self.regularizer)(x)
        x = BatchNormalization(gamma_initializer = 'zeros')(x)
        if projection:
            shortcut = Conv2D(num_filters,
                              (1, 1),
                              padding = "same",
                              use_bias = False,
                              kernel_initializer = self.initializer,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization()(shortcut)
        elif strides != (1, 1):
            shortcut = Conv2D(num_filters,
                              (1, 1),
                              strides = strides,
                              padding = "same",
                              use_bias = False,
                              kernel_initializer = self.initializer,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        y = tf.nn.relu(x)
        return y

    def build_model (self):
        x_in = Input(shape = (None, None, 3), name = "input")

        # The first conv layer.
        x = Conv2D(16,
                   (3, 3),
                   strides=(1, 1),
                   name='conv0',
                   padding='same',
                   use_bias=False,
                   kernel_initializer = self.initializer,
                   kernel_regularizer = self.regularizer) (x_in)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        # Residual blocks
        for i in range (3):
            if i == 0:
                x = self.res_block(x, 16, projection = True)
            else:
                x = self.res_block(x, 16)

        for i in range (3):
            if i == 0:
                x = self.res_block(x, 32, strides = (2, 2))
            else:
                x = self.res_block(x, 32)

        for i in range (3):
            if i == 0:
                x = self.res_block(x, 64, strides = (2, 2))
            else:
                x = self.res_block(x, 64)

        # The final average pooling layer and fully-connected layer.
        x = GlobalAveragePooling2D()(x)
        y = Dense(10, activation = 'softmax', name='fully_connected',
                  kernel_initializer = self.initializer,
                  kernel_regularizer = self.regularizer,
                  bias_regularizer = self.regularizer)(x)
        return Model(x_in, y, name = "resnet20")

class resnet50 ():
    def __init__ (self, weight_decay):
        self.weight_decay = weight_decay
        self.batch_norm_momentum = 0.9
        self.batch_norm_epsilon = 1e-5

    def res_block (self, input_tensor, num_filters, strides = (1, 1), projection = False):
        x = Conv2D(num_filters, (1, 1), padding = "same", use_bias = False,
                   kernel_regularizer = self.regularizer)(input_tensor)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)

        x = Conv2D(num_filters, (3, 3), strides = strides, padding = "same", use_bias = False,
                   kernel_regularizer = self.regularizer)(x)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)

        x = Conv2D(num_filters * 4, (1, 1), padding = "same", use_bias = False,
                   kernel_regularizer = self.regularizer)(x)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        if projection:
            shortcut = Conv2D(num_filters * 4, (1, 1), padding = "same", use_bias = False,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                          epsilon = self.batch_norm_epsilon)(shortcut)
        elif strides != (1, 1):
            shortcut = tf.nn.avg_pool2d(input_tensor, ksize = (2, 2), strides = (2, 2), padding = "SAME")
            shortcut = Conv2D(num_filters * 4, (1, 1), padding = "same", use_bias = False,
                              kernel_regularizer = self.regularizer)(shortcut)
            shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                          epsilon = self.batch_norm_epsilon)(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        y = tf.nn.relu(x)
        return y

    def build_model (self):
        self.regularizer = l2(self.weight_decay)

        x_in = Input(shape = (224, 224, 3), name = "input")

        # The first conv layer.
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv0', padding='same', use_bias=False,
                   kernel_regularizer = self.regularizer)(x_in)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=(3, 3), strides=(2, 2), padding='SAME')

        # Residual blocks
        for i in range (3):
            if i == 0:
                x = self.res_block(x, 64, projection = True)
            else:
                x = self.res_block(x, 64)

        for i in range (4):
            if i == 0:
                x = self.res_block(x, 128, strides = (2, 2))
            else:
                x = self.res_block(x, 128)

        for i in range (6):
            if i == 0:
                x = self.res_block(x, 256, strides = (2, 2))
            else:
                x = self.res_block(x, 256)

        for i in range (3):
            if i == 0:
                x = self.res_block(x, 512, strides = (2, 2))
            else:
                x = self.res_block(x, 512)

        # The final average pooling layer and fully-connected layer.
        x = GlobalAveragePooling2D()(x)
        y = Dense(1000, name='fully_connected', activation='softmax', use_bias=False,
                  kernel_regularizer = self.regularizer)(x)
        return Model(x_in, y, name = "resnet50")

class wideresnet16 ():
    def __init__ (self, weight_decay):
        self.weight_decay = weight_decay
        self.batch_norm_momentum = 0.99
        self.batch_norm_epsilon = 1e-5

    def res_block (self, input_tensor, num_filters, strides = (1, 1), projection = False):
        x = Conv2D(num_filters,
                   (3, 3),
                   strides = strides,
                   padding = "same",
                   use_bias = False,
                   kernel_regularizer = self.regularizer)(input_tensor)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)
        x = Dropout(0.3)(x)

        x = Conv2D(num_filters,
                   (3, 3),
                   padding = "same",
                   use_bias = False,
                   kernel_regularizer = self.regularizer)(x)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        if projection:
            shortcut = Conv2D(num_filters,
                              (1, 1),
                              padding = "same",
                              use_bias = False,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                          epsilon = self.batch_norm_epsilon)(shortcut)
        elif strides != (1, 1):
            shortcut = Conv2D(num_filters,
                              (1, 1),
                              strides = strides,
                              padding = "same",
                              use_bias = False,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                          epsilon = self.batch_norm_epsilon)(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        y = tf.nn.relu(x)
        return y

    def build_model (self):
        self.regularizer = l2(self.weight_decay)
        d = 16
        k = 8
        rounds = int((d - 4) / 6)

        x_in = Input(shape = (None, None, 3), name = "input")

        # The first conv layer.
        x = Conv2D(16,
                   (3, 3),
                   strides=(1, 1),
                   name='conv0',
                   padding='same',
                   use_bias=False,
                   kernel_regularizer = self.regularizer) (x_in)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)

        # Residual blocks
        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 16 * k, projection = True)
            else:
                x = self.res_block(x, 16 * k)

        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 32 * k, strides = (2, 2))
            else:
                x = self.res_block(x, 32 * k)

        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 64 * k, strides = (2, 2))
            else:
                x = self.res_block(x, 64 * k)

        # The final average pooling layer and fully-connected layer.
        x = GlobalAveragePooling2D()(x)
        y = Dense(10, activation = 'softmax', name='fully_connected', kernel_regularizer = self.regularizer)(x)
        return Model(x_in, y, name = "wideresnet16-8")

class wideresnet28 ():
    def __init__ (self, weight_decay):
        self.weight_decay = weight_decay
        self.batch_norm_momentum = 0.99
        self.batch_norm_epsilon = 1e-5

    def res_block (self, input_tensor, num_filters, strides = (1, 1), projection = False):
        x = Conv2D(num_filters,
                   (3, 3),
                   strides = strides,
                   padding = "same",
                   use_bias = False,
                   kernel_regularizer = self.regularizer)(input_tensor)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)
        x = Dropout(0.3)(x)

        x = Conv2D(num_filters,
                   (3, 3),
                   padding = "same",
                   use_bias = False,
                   kernel_regularizer = self.regularizer)(x)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               gamma_initializer = 'zeros',
                               epsilon = self.batch_norm_epsilon)(x)
        if projection:
            shortcut = Conv2D(num_filters,
                              (1, 1),
                              padding = "same",
                              use_bias = False,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                          epsilon = self.batch_norm_epsilon)(shortcut)
        elif strides != (1, 1):
            shortcut = Conv2D(num_filters,
                              (1, 1),
                              strides = strides,
                              padding = "same",
                              use_bias = False,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                          epsilon = self.batch_norm_epsilon)(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        y = tf.nn.relu(x)
        return y

    def build_model (self):
        self.regularizer = l2(self.weight_decay)
        d = 28
        k = 10
        rounds = int((d - 4) / 6)

        x_in = Input(shape = (None, None, 3), name = "input")

        # The first conv layer.
        x = Conv2D(16,
                   (3, 3),
                   strides=(1, 1),
                   name='conv0',
                   padding='same',
                   use_bias=False,
                   kernel_regularizer = self.regularizer) (x_in)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)

        # Residual blocks
        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 16 * k, projection = True)
            else:
                x = self.res_block(x, 16 * k)

        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 32 * k, strides = (2, 2))
            else:
                x = self.res_block(x, 32 * k)

        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 64 * k, strides = (2, 2))
            else:
                x = self.res_block(x, 64 * k)

        # The final average pooling layer and fully-connected layer.
        x = GlobalAveragePooling2D()(x)
        y = Dense(100, activation = 'softmax', name='fully_connected',
                  kernel_regularizer = self.regularizer,
                  bias_regularizer = self.regularizer)(x)
        return Model(x_in, y, name = "wideresnet28-10")

class cnn ():
    def __init__ (self, weight_decay, num_classes):
        self.weight_decay = weight_decay
        self.regularizer = l2(self.weight_decay)
        self.num_classes = num_classes

    def build_model (self):
        x_in = Input(shape=(28, 28, 1), name="input")

        conv1 = Conv2D(
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation='relu')(x_in)

        pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(conv1)

        conv2 = Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation='relu')(pool1)

        pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(conv2)

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        dense = tf.keras.layers.Dense(units=2048, activation=tf.nn.relu)(pool2_flat)

        logits = tf.keras.layers.Dense(units=self.num_classes)(dense)

        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        y = predictions["probabilities"]
        return Model(x_in, y, name = "cnn")

class Network ():
    def __init__ (self, input_length, weight_decay, num_classes):
        self.sample_length = input_length
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.batch_norm_momentum = 0.99
        self.batch_norm_epsilon = 1e-6
        self.filter_size = 512
        self.dense_width = 1024
        self.kernel_size = 5
        self.num_conv_layers = 10
        self.num_full_layers = 2
        self.initializer = tf.keras.initializers.GlorotNormal(seed = int(time.time()))
        print ("Initializing a simple feed-forward network...")

    def build_model2 (self):
        self.regularizer = l2(self.weight_decay)
        x_in = Input(shape = (self.sample_length), name = 'input')
        x = Dense(self.dense_width, kernel_regularizer = self.regularizer)(x_in)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        size = self.filter_size
        for i in range (self.num_full_layers):
            x = Dense(size, kernel_regularizer = self.regularizer)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU() (x)
        y = Dense(self.num_classes, 
                  kernel_regularizer = self.regularizer,
                  activation = 'softmax')(x)
        return Model(x_in, y, name = "oxynet")

    def build_model (self):
        filter_size = 64
        self.regularizer = l2(self.weight_decay)
        x_in = Input(shape = (self.sample_length, 1), name = 'input')
        #x = Conv1D(filters = self.filter_size,
        x = Conv1D(filters = filter_size,
                   #kernel_size = self.kernel_size,
                   kernel_size = 64,
                   strides = 1,
                   kernel_regularizer = self.regularizer,
                   padding = "same")(x_in)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        for i in range (self.num_conv_layers):
            x = Conv1D(filters = self.filter_size,
            #x = Conv1D(filters = filter_size,
                       kernel_size = self.kernel_size,
                       strides = 1,
                       kernel_regularizer = self.regularizer,
                       padding = "same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if i % 3 == 0:
                x = MaxPooling1D(pool_size = 2)(x)
        #x = GlobalAveragePooling1D()(x)
        x = tf.reshape(x, [-1, np.prod(x.shape[1:])])

        for i in range (self.num_full_layers):
            x = Dense(self.dense_width, kernel_regularizer = self.regularizer)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

        y = Dense(self.num_classes,
                  kernel_regularizer = self.regularizer,
                  activation = 'softmax')(x)
        return Model(x_in, y, name = "oxynet")
