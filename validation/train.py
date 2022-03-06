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
        self.model = model.build_model()
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.train_loss = tf.keras.losses.CategoricalCrossentropy()
        self.valid_acc = tf.keras.metrics.Accuracy()
        self.valid_top5acc = tf.keras.metrics.TopKCategoricalAccuracy()
        self.valid_loss = tf.keras.losses.MeanAbsoluteError()
        optimizer = SGD(learning_rate = self.min_lr, momentum = 0.9)
        #optimizer = Adam(learning_rate = self.min_lr)

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
            predicts = self.checkpoint.model(data, training = True)
            loss = self.cross_entropy_batch(labels, predicts)
            regularization_losses = self.model.losses
            total_loss = tf.add_n(regularization_losses + [loss])
        gradients = tape.gradient(total_loss, self.checkpoint.model.trainable_variables)
        #gradients = tape.gradient(loss, self.checkpoint.model.trainable_variables)
        global_grads = comm.average_grads(self.checkpoint.model, gradients)
        self.checkpoint.optimizer.apply_gradients(zip(global_grads, self.checkpoint.model.trainable_variables))
        return loss, predicts

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
                loss, predicts = self.train_step(images, labels)
                loss_mean(loss)
                '''
                if i == 0:
                    for k in range (100):
                        print ("in[%2d]: %10.6f predicts[%2d]: %f label[%2d]: %f\n" %(k, images[0][k], k, predicts[0][k], k, labels[0][k]))
                '''
            train_loss = loss_mean.result().numpy()
            global_loss = self.comm.allreduce(train_loss, op = MPI.SUM) / self.size
            global_loss = train_loss
            loss_mean.reset_states()

            # Evaluation.
            global_acc, global_top5acc = self.evaluate()
            if self.rank == 0:
                print ("Epoch " + str(epoch_id) +
                       " lr: " + str(self.checkpoint.optimizer.lr.numpy()) +
                       " training loss = " + str(global_loss),
                       " validation acc = " + str(global_acc),
                       " top5 acc = " + str(global_top5acc))
                f = open("loss.txt", "a")
                f.write(str(global_loss) + "\n")
                f.close()
                f = open("acc.txt", "a")
                f.write(str(global_acc) + "\n")
                f.close()
                f = open("top5acc.txt", "a")
                f.write(str(global_top5acc) + "\n")
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

    def evaluate (self):
        valid_dataset = self.dataset.valid_dataset()
        num_iterations = int(self.dataset.num_valid_batches)
        for i in tqdm(range(num_iterations), ascii=True):
            data, labels = valid_dataset.next()
            predicts = self.checkpoint.model(data)
            results = tf.argmax(predicts, 1).numpy()
            label = tf.argmax(labels[0]).numpy()
            if label == 0:
                name = "time_0s.txt"
            elif label == 1:
                name = "time_10s.txt"
            elif label == 2:
                name = "time_30s.txt"
            else:
                name = "time_60s.txt"
            f = open(name, "a")
            for j in range (len(results)):
                f.write("%3d\n" %(results[j]))
            f.close()
        return

    def cross_entropy_batch (self, label, prediction):
        cross_entropy = self.train_loss(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def record (self):
        valid_dataset = self.dataset.valid_dataset()
        num_iterations = int(self.dataset.num_valid_batches * self.size)
        self.valid_acc.reset_states()
        valid_loss_mean = Mean()
        latents = []
        f = open("latent.txt", "a")
        f2 = open("index.txt", "a")
        for i in tqdm(range(num_iterations), ascii=True):
            data, labels, indices = valid_dataset.next()
            bottleneck, predicts = self.checkpoint.model(data)
            latents.append(bottleneck.numpy())
            for j in range (self.dataset.valid_batch_size):
                latent = latents[i][j]
                for k in range(len(latent)):
                    f.write("%10.6f," %(latent[k]))
                f.write("\n")

                index = indices.numpy()[j]
                f2.write("%2d\n" %(index))
        f.close()
        f2.close()

    def cluster (self):
        print ("Clustering...")

        # Read latent file.
        f = open("latent.txt", "r")
        lines = f.readlines()
        f.close()
        latent = []
        for i in range (len(lines)):
            line = lines[i].split('\n')[0]
            tokens = line.split(',')
            latent.append(tokens[0:-1])
        latent = np.array(latent).astype(np.float)

        # Read label file.
        f = open("index.txt", "r")
        lines = f.readlines()
        f.close()
        index = []
        for i in range (len(lines)):
            line = lines[i].split('\n')[0]
            token = line.split('\n')
            index.append(token)
        index = np.array(index).astype(np.int)

        # Run clustering...
        kmeans = KMeans(n_clusters = 2, random_state = 0).fit(latent)

        # Flip the value to compare with the input label that starts with 1.
        num_correct = 0
        f = open("cluster.txt", "a")
        for i in range(len(kmeans.labels_)):
            value = kmeans.labels_[i]
            if index[i] == value:
                num_correct += 1
            f.write("%2d\n" %(value))
        f.close()

        ratio = 100.0 * num_correct / len(kmeans.labels_)
        if ratio < 50.0:
            ratio = 100.0 - ratio
        print ("hit ratio: %10.6f out of %d data\n" %(ratio, len(kmeans.labels_) ))
