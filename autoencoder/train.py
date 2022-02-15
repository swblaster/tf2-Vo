import time
import math
import numpy as np
import tensorflow as tf
import argparse
import comm
from mpi4py import MPI
from tqdm import tqdm
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans

class Trainer:
    def __init__ (self, model, dataset, num_epochs, min_lr, max_lr,
                  weight_decay, num_classes, decay_epochs, do_checkpoint = False):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.dataset = dataset
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_epochs = num_epochs
        self.decay_epochs = decay_epochs
        self.lr_decay_factor = 10
        self.do_checkpoint = do_checkpoint
        self.model = model.autoencoder()
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.train_loss = tf.keras.losses.MeanAbsoluteError()
        #optimizer = SGD(learning_rate = self.min_lr, momentum = 0.9)
        optimizer = Adam(learning_rate = self.min_lr)

        # Assign additional parameters for customized parameter update.
        self.params = []
        self.bn = []
        for i in range (len(self.model.trainable_variables)):
            param = self.model.trainable_variables[i]
            param_type = param.name.split("/")[1].split(":")[0]
            if param_type == "beta" or param_type == "gamma":
                self.bn.append("bn")
            elif param_type == "bias":
                self.bn.append("bias")
            elif param_type == "kernel":
                self.bn.append("kernel")
            else:
                self.bn.append("etc")

            self.params.append(i)

        # Checkpointing.
        self.checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0),
                                              model = self.model,
                                              optimizer = optimizer)
        checkpoint_dir = "./check_" + str(self.rank)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint = self.checkpoint,
                                                             directory = checkpoint_dir,
                                                             max_to_keep = 3)

        self.resume()
        if self.rank == 0:
            self.model.summary()

    def resume (self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            if self.rank == 0:
                print ("Model restored from checkpoint at epoch " + str(self.checkpoint.epoch.numpy()))

    #@tf.function
    def train_step (self, data, labels):
        with tf.GradientTape() as tape:
            hidden, predicts = self.checkpoint.model(data, training = True)
            loss = self.cross_entropy_batch(labels, predicts)
        gradients = tape.gradient(loss, self.checkpoint.model.trainable_variables)
        global_grads = comm.average_grads(self.checkpoint.model, gradients)
        self.checkpoint.optimizer.apply_gradients(zip(global_grads, self.checkpoint.model.trainable_variables))
        return loss, predicts, hidden

    def train (self):
        train_dataset = self.dataset.train_dataset()

        # A checkpoint has epoch that is the first epoch after resume.
        start_epoch = self.checkpoint.epoch.numpy()
        if start_epoch == 0:
            comm.broadcast_model(self.checkpoint.model)
            print ("Starting from the scratch!\n")
        else:
            print ("Resuming from epoch " + str(start_epoch))

        for epoch_id in range (start_epoch, self.num_epochs):
            self.dataset.shuffle()
            loss_mean = Mean()

            # Training loop.
            for i in tqdm(range(self.dataset.num_train_batches), ascii=True):
                images, labels = train_dataset.next()
                loss, predicts, hidden = self.train_step(images, labels)
                loss_mean(loss)
            train_loss = loss_mean.result().numpy()
            global_loss = self.comm.allreduce(train_loss, op = MPI.SUM) / self.size
            global_loss = train_loss
            loss_mean.reset_states()

            if self.rank == 0:
                print ("Epoch %3d lr: %8.6f loss: %f\n" %(epoch_id, self.checkpoint.optimizer.lr.numpy(), global_loss))
                f = open("loss.txt", "a")
                f.write(str(global_loss) + "\n")
                f.close()

            # Adjust the learning rate.
            if epoch_id in self.decay_epochs:
                lr_decay = 1 / self.lr_decay_factor
                self.checkpoint.optimizer.lr.assign(self.checkpoint.optimizer.lr * lr_decay)
                if self.rank == 0:
                    print ("LR decay to " + str(self.checkpoint.optimizer.lr))

            # Checkpointing.
            self.checkpoint.epoch.assign_add(1)
            if self.do_checkpoint == True:
                self.checkpoint_manager.save()
            self.comm.Barrier()

    def cross_entropy_batch (self, label, prediction):
        cross_entropy = self.train_loss(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def record (self):
        train_dataset = self.dataset.train_dataset()

        # A checkpoint has epoch that is the first epoch after resume.
        start_epoch = self.checkpoint.epoch.numpy()
        if start_epoch == 0:
            comm.broadcast_model(self.checkpoint.model)
            print ("Starting from the scratch!\n")
        else:
            print ("Resuming from epoch " + str(start_epoch))

        step = 0
        sigma = 50
        for i in tqdm(range(self.dataset.num_train_batches), ascii=True):
            images, labels = train_dataset.next()
            loss, predicts, hidden = self.train_step(images, labels)
            name = "hidden_" + str(sigma) + ".txt"
            f = open(name, "a")
            hidden_data = hidden.numpy().flatten()
            for j in range (len(hidden_data)):
                f.write("%18.16f\n" %(hidden_data[j]))
            f.close()
            step += 1
            if step == 20:
                step = 0
                sigma += 50
        self.comm.Barrier()
