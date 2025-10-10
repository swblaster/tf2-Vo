'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2023/08/09
Sunwoo Lee, Ph.D.
<sunwool@inha.ac.kr>
'''
from mpi4py import MPI
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

class GSAM:
    def __init__ (self, num_classes, model):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        self.optimizer = SGD(momentum = 0.9)
        self.r = 0.1
        self.a = 0.8
        self.num_t_params = len(model.trainable_variables)
        self.num_nt_params = len(model.non_trainable_variables)
        self.weight_index = []
        for i in range (self.num_t_params):
            if len(model.trainable_variables[i].shape) > 1:
                self.weight_index.append(i)
        self.num_weights = len(self.weight_index)
        self.perturb_list = []
        self.num_to_perturb = 0
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1)
        if self.rank == 0:
            print ("GSAM is the solver!")

    @tf.function
    def cross_entropy_batch(self, label, prediction):
        cross_entropy = self.loss_object(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def train_step (self, checkpoint, data, label):
        # Keep the original model parameters.
        orig_params = []
        for i in range (self.num_weights):
            orig_params.append(tf.identity(checkpoint.model.trainable_variables[self.weight_index[i]]))

        # Calculate g1.
        with tf.GradientTape() as tape:
            prediction = checkpoint.model(data, training = True)
            loss = self.cross_entropy_batch(label, prediction)
            regularization_losses = checkpoint.model.losses
            total_loss = tf.add_n(regularization_losses + [loss])
        g1 = tape.gradient(total_loss, checkpoint.model.trainable_variables)

        global_g1 = []
        norms = []
        for i in range (self.num_weights):
            g = g1[self.weight_index[i]].numpy()
            g = self.comm.allreduce(g, op = MPI.SUM) / self.size
            norms.append(np.linalg.norm(g.flatten()))
            global_g1.append(g)
        global_norm = np.linalg.norm(norms)

        # Add it to the model.
        for i in range (self.num_weights):
            param = checkpoint.model.trainable_variables[self.weight_index[i]]
            update = self.r / global_norm * global_g1[i]
            checkpoint.model.trainable_variables[self.weight_index[i]].assign(tf.math.add(param, update))

        # Calculate g2.
        with tf.GradientTape() as tape:
            prediction = checkpoint.model(data, training = True)
            loss = self.cross_entropy_batch(label, prediction)
            regularization_losses = checkpoint.model.losses
            total_loss = tf.add_n(regularization_losses + [loss])
        g2 = tape.gradient(total_loss, checkpoint.model.trainable_variables)
        global_g2 = []
        for i in range (len(g2)):
            g = g2[i].numpy()
            g = self.comm.allreduce(g, op = MPI.SUM) / self.size
            global_g2.append(g)

        g = []
        offset = 0
        for i in range (self.num_t_params):
            if i in self.weight_index:
                g.append((1 - self.a) * global_g1[offset] + self.a * global_g2[i])
                offset += 1
            else:
                g.append(global_g2[i])

        # Rollback the model.
        for i in range (self.num_weights):
            checkpoint.model.trainable_variables[self.weight_index[i]].assign(orig_params[i])

        # Apply the new grads to the original model.
        checkpoint.optimizer.apply_gradients(zip(g, checkpoint.model.trainable_variables))
        return loss, g

    def average_model (self, checkpoint, epoch_id):
        # non-trainable parameters (BN statistics)
        for i in range (self.num_nt_params):
            local_param = checkpoint.model.non_trainable_variables[i]
            averaged_param = self.comm.allreduce(local_param, op = MPI.SUM) / self.size
            checkpoint.model.non_trainable_variables[i].assign(averaged_param)
