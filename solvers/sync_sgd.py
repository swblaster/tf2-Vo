'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2023/03/17
Sunwoo Lee, Ph.D.
<sunwool@inha.ac.kr>
'''
from mpi4py import MPI
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

class SyncSGD:
    def __init__ (self, num_classes, model):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        #self.optimizer = SGD(momentum = 0.9)
        self.optimizer = Adam()
        self.num_t_params = len(model.trainable_variables)
        self.num_nt_params = len(model.non_trainable_variables)
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1)
        if self.rank == 0:
            print ("Synchronous SGD is the solver!")

    @tf.function
    def cross_entropy_batch(self, label, prediction):
        cross_entropy = self.loss_object(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def train_step (self, checkpoint, data, label):
        with tf.GradientTape() as tape:
            prediction = checkpoint.model(data, training = True)
            loss = self.cross_entropy_batch(label, prediction)
            regularization_losses = checkpoint.model.losses
            total_loss = tf.add_n(regularization_losses + [loss])
        grads = tape.gradient(total_loss, checkpoint.model.trainable_variables)
        checkpoint.optimizer.apply_gradients(zip(grads, checkpoint.model.trainable_variables))
        return loss, grads

    def average_model (self, checkpoint, epoch_id):
        # trainable parameters
        for i in range (self.num_t_params):
            local_param = checkpoint.model.trainable_variables[i]
            averaged_param = self.comm.allreduce(local_param, op = MPI.SUM) / self.size
            checkpoint.model.trainable_variables[i].assign(averaged_param)

        '''
        # non-trainable parameters (BN statistics)
        for i in range (self.num_nt_params):
            local_param = checkpoint.model.non_trainable_variables[i]
            averaged_param = self.comm.allreduce(local_param, op = MPI.SUM) / self.size
            checkpoint.model.non_trainable_variables[i].assign(averaged_param)
        '''
