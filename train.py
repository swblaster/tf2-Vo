'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2023/03/17
Sunwoo Lee, Ph.D.
<sunwool@inha.ac.kr>
'''
import time
import math
import random
import numpy as np
import tensorflow as tf
import argparse
from mpi4py import MPI
from tqdm import tqdm
from tensorflow.keras.metrics import Mean

class framework:
    def __init__ (self, model, dataset, solver, **kargs):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.dataset = dataset
        self.solver = solver
        self.num_epochs = kargs["num_epochs"]
        self.min_lr = kargs["min_lr"]
        self.max_lr = kargs["max_lr"]
        self.decay_epochs = kargs["decay_epochs"]
        self.do_checkpoint = kargs["do_checkpoint"]
        self.num_classes = kargs["num_classes"]
        self.warmup_epochs = 0
        self.lr_decay_factor = 10
        if self.num_classes == 1:
            self.valid_acc = tf.keras.metrics.BinaryAccuracy()
        else:
            self.valid_acc = tf.keras.metrics.Accuracy()

        self.checkpoint = tf.train.Checkpoint(model = model, optimizer = self.solver.optimizer)
        self.checkpoint.optimizer.learning_rate.assign(self.min_lr)
        checkpoint_dir = "./checkpoint_" + str(self.rank)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint = self.checkpoint, directory = checkpoint_dir, max_to_keep = 3)
        if self.rank == 0:
            self.checkpoint.model.summary()

        # Resume if any checkpoints are in the current directory.
        self.resume()

    def resume (self):
        self.epoch_id = 0
        if self.checkpoint_manager.latest_checkpoint:
            self.epoch_id = int(self.checkpoint_manager.latest_checkpoint.split(sep='ckpt-')[-1]) - 1
            if self.rank == 0:
                print ("Resuming the training from epoch %3d\n" % (self.epoch_id))
            if self.checkpoint_manager.latest_checkpoint:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                test_dataset = self.dataset.test_dataset()
                self.test(test_dataset)
                self.comm.Barrier()
                exit()

    def train (self):
        train_dataset = self.dataset.train_dataset()
        valid_dataset = self.dataset.valid_dataset()

        # Calculate the warmup lr increase.
        warmup_step_lr = 0
        if self.warmup_epochs > 0:
            num_warmup_steps = self.warmup_epochs * self.average_interval
            warmup_step_lr = (self.max_lr - self.min_lr) / num_warmup_steps

        # Broadcast the parameters from rank 0 at the first epoch.
        start_epoch = self.epoch_id
        if start_epoch == 0:
            self.broadcast_model()

        for epoch_id in range (start_epoch, self.num_epochs):
            lossmean = Mean()
            self.dataset.shuffle()

            # LR decay
            if epoch_id in self.decay_epochs:
                lr_decay = 1 / self.lr_decay_factor
                self.checkpoint.optimizer.learning_rate.assign(self.checkpoint.optimizer.learning_rate * lr_decay)

            # Training loop.
            lossmean.reset_state()
            for j in tqdm(range(self.dataset.num_train_batches), ascii=True):
                if epoch_id < self.warmup_epochs:
                    self.checkpoint.optimizer.learning_rate.assign(self.checkpoint.optimizer.learning_rate + warmup_step_lr)
                images, labels = train_dataset.next()
                loss, grads = self.solver.train_step(self.checkpoint, images, labels)
                self.solver.average_model(self.checkpoint, epoch_id)
                lossmean(loss)

            # Collect the global training results (loss and accuracy).
            local_loss = lossmean.result().numpy()
            global_loss = self.comm.allreduce(local_loss, op = MPI.SUM) / self.size

            # Collect the global validation accuracy.
            local_acc = self.evaluate(epoch_id, valid_dataset)
            global_acc = self.comm.allreduce(local_acc, op = MPI.MAX)

            # Checkpointing
            if self.do_checkpoint == True:
                self.checkpoint_manager.save()

            if self.rank == 0:
                print ("Epoch " + str(epoch_id) +
                       " lr: " + str(self.checkpoint.optimizer.learning_rate.numpy()) +
                       " validation acc = " + str(global_acc) +
                       " training loss = " + str(global_loss))
                f = open("acc.txt", "a")
                f.write(str(global_acc) + "\n")
                f.close()
                f = open("loss.txt", "a")
                f.write(str(global_loss) + "\n")
                f.close()

            # Exit
            if global_acc > 0.95:
                self.comm.Barrier()
                exit()

    def evaluate (self, epoch_id, valid_dataset):
        self.valid_acc.reset_state()
        for i in tqdm(range(self.dataset.num_valid_batches), ascii=True):
            data, label = valid_dataset.next()
            predicts = self.checkpoint.model(data)
            if len(label.shape) == 1:
                self.valid_acc(label, predicts)
            else:
                self.valid_acc(tf.argmax(label, 1), tf.argmax(predicts, 1))
        accuracy = self.valid_acc.result().numpy()
        return accuracy

    def test (self, test_dataset):
        data = test_dataset.next()
        predicts = self.checkpoint.model(data)
        predicts = tf.argmax(predicts, 1)
        for i in range (len(predicts)):
            class_id = predicts[i]
            candidates, e_sigmas, e_cuts, t_cuts = self.dataset.get_info(class_id)
            sample_id =  i // 10
            observation_id = i % 10
            for i in range (len(candidates)):
                print ("Sample: %2d Observation: %2d Class ID: %2d index: %3d e_sigma: %2d e_cut: %2d t_cut: %2d" %(sample_id, observation_id, class_id, candidates[i], e_sigmas[i], e_cuts[i], t_cuts[i]))

    def broadcast_model (self):
        num_params = len(self.checkpoint.model.trainable_variables)
        for i in range (num_params):
            param = self.checkpoint.model.trainable_variables[i]
            param = self.comm.bcast(param, root=0)
            self.checkpoint.model.trainable_variables[i].assign(param)
