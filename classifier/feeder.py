'''
Sunwoo Lee Ph.D.
Postdoctoral Researcher
University of Southern California
<sunwool@usc.edu>
'''

import random
import time
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import h5py
from mpi4py import MPI
import tensorflow as tf
import csv

def dense_to_one_hot(labels_dense, num_classes=100):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=float)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

class Feeder:
    def __init__ (self, input_path, batch_size, input_length, num_classes):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.input_path = input_path
        self.train_batch_size = batch_size
        self.valid_batch_size = 10
        self.input_length = input_length
        self.num_classes = num_classes

        # Read all the files.
        files = [f for f in listdir(self.input_path) if isfile(join(self.input_path, f))]
        data = []
        sample_labels = []
        for i in range(len(files)):
            path = self.input_path + "/" + files[i]
            f = open(path, "r")
            lines = f.readlines()
            sample = []
            for j in range (len(lines)):
                line = lines[j].split('\n')
                value = float(line[0])
                sample.append(value)
            data.append(sample)

            tokens = files[i].split('_')
            sigma = tokens[1].split('.')[0]
            label = np.zeros((1000))
            for j in range (1000):
                label[j] = float((int(sigma) - 300) // 50.0)
            sample_labels.append(label)
            print ("label: %f shape: %d\n" %(label[0], len(sample)))
        samples = np.array(data)
        labels = np.array(sample_labels)
        self.samples = np.reshape(samples, (5000, 200))
        self.samples = self.samples[:,:self.input_length]
        labels = np.reshape(labels, (5000))
        self.labels = np.zeros((5000, 5))
        for i in range (5000):
            index = int(labels[i])
            self.labels[i][index] = 1

        self.num_train_samples = 4500
        self.num_train_batches = self.num_train_samples // (self.size * self.train_batch_size)
        self.num_local_train_samples = self.num_train_batches * self.train_batch_size
        self.local_train_samples_offset = self.rank * self.num_local_train_samples
        self.num_valid_samples = 500
        self.num_valid_batches = self.num_valid_samples // self.valid_batch_size

        '''
        # First, normalize the data within each sample.
        for i in range (4000):
            max_element = max(self.samples[i])
            min_element = min(self.samples[i])
            self.samples[i] = (self.samples[i] - min_element) / (max_element - min_element)
        '''
        # Then, standardize each point across all the samples.
        '''
        per_pixel_mean = np.array(self.samples).astype(np.float32).mean(axis=0)
        per_pixel_std = np.array(self.samples).astype(np.float32).std(axis=0)
        self.samples = np.subtract(self.samples, per_pixel_mean)
        self.samples = np.divide(self.samples, per_pixel_std)
        '''

        #'''
        per_pixel_max = np.amax(self.samples, axis=0)
        per_pixel_min = np.amin(self.samples, axis=0)
        self.samples = (self.samples - per_pixel_min) / (per_pixel_max - per_pixel_min)
        #'''

        train = []
        train_label = []
        valid = []
        valid_label = []
        offset = 0
        for i in range (5):
            for j in range (1000):
                if j < 900:
                    train.append(self.samples[offset])
                    train_label.append(self.labels[offset])
                else:
                    valid.append(self.samples[offset])
                    valid_label.append(self.labels[offset])
                offset += 1
        self.train_samples = train
        self.train_labels = train_label
        self.valid_samples = valid
        self.valid_labels = valid_label
        print ("Local batch size: " + str(self.train_batch_size))
        print ("Number of training samples: " + str(self.num_train_samples))
        print ("Number of training batches: " + str(self.num_train_batches))
        print ("Number of validation samples: " + str(self.num_valid_samples))
        print ("Number of validation batches: " + str(self.num_valid_batches))

    def shuffle (self):
        self.shuffled_sample_index = np.arange(self.num_train_samples)
        random.seed(time.time())
        random.shuffle(self.shuffled_sample_index)
        self.comm.Bcast(self.shuffled_sample_index, root = 0) 

    def read_train_sample (self, sample_id):
        offset = self.local_train_samples_offset + sample_id
        index = self.shuffled_sample_index[offset]
        data = self.train_samples[index]
        data = np.reshape(data, [self.input_length, 1])
        label = self.train_labels[index]
        return data, label

    def tf_read_train_sample (self, sample_id):
        data, label = tf.py_function(self.read_train_sample, inp=[sample_id], Tout=[tf.float32, tf.float32])
        return data, label

    def train_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_local_train_samples))
        dataset = dataset.map(self.tf_read_train_sample)
        dataset = dataset.batch(self.train_batch_size)
        dataset = dataset.repeat()
        return dataset.__iter__()

    def read_valid_sample (self, sample_id):
        index = sample_id
        data = self.valid_samples[index]
        data = np.reshape(data, [self.input_length, 1])
        label = self.valid_labels[index]
        return data, label

    def tf_read_valid_sample (self, sample_id):
        data, label = tf.py_function(self.read_valid_sample, inp=[sample_id], Tout=[tf.float32, tf.float32])
        return data, label

    def valid_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_valid_samples))
        dataset = dataset.map(self.tf_read_valid_sample)
        dataset = dataset.batch(self.valid_batch_size)
        dataset = dataset.repeat()
        return dataset.__iter__()
