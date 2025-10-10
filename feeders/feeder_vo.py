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
from sklearn.cluster import DBSCAN
import csv

class Vo:
    def __init__ (self, batch_size, input_length, cluster = -1):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.train_batch_size = batch_size
        self.valid_batch_size = 100
        self.input_length = input_length
        self.cluster = cluster
        self.standardize = True

        # Read the test data.
        f = open("test.bin", "r")
        lines = f.readlines()
        test_data = []
        for line in lines:
            data = float(line.split("\n")[0])
            test_data.append(data)
        self.test_data = np.array(test_data).reshape((80, 100))
        f.close()

        # Read the training file.
        #e_sigmas = np.arange(1, 11)
        #e_cuts = np.arange(7, 17)
        e_sigmas = np.arange(1, 16)
        e_cuts = np.arange(7, 22)
        t_cuts = np.arange(4, 24)

        dsets = []
        f = h5py.File("data_es1-16_ec7-22_tc4-24_alpha1.h5", "r")
        '''
        num_dsets = len(self.e_cuts) * len(self.t_cuts)
        for i in range (num_dsets):
            dset = np.array(f["/" + str(i)])
            dsets.append(dset)
        '''
        self.e_sigmas = []
        self.e_cuts = []
        self.t_cuts = []
        num_blocks = 0
        for i in range (len(e_cuts)):
            if i > 10:
                continue
            e_sigma = e_sigmas[i]
            e_cut = e_cuts[i]
            for j in range (len(t_cuts)):
                t_cut = t_cuts[j]
                index = i * len(t_cuts) + j
                num_blocks += 1
                self.e_sigmas.append(e_sigma)
                self.e_cuts.append(e_cut)
                self.t_cuts.append(t_cut)
                dset = np.array(f["/" + str(index)])
                dsets.append(dset)
        self.dsets = np.array(dsets)
        f.close()

        dimensions = self.dsets.shape
        dsets = np.reshape(dsets, (num_blocks, dimensions[-2], dimensions[-1]))

        # Prepare the labels.
        dsets = np.mean(dsets, axis=1)
        clustering = DBSCAN(eps=2.0, min_samples=1).fit(dsets)
        labels = clustering.labels_
        self.labels = labels.flatten()
        if self.rank == 0:
            f = open("label.txt", "w")
            for i in range (len(self.labels)):
                f.write("%d\n" %(self.labels[i]))
            f.close()
        self.num_classes = len(np.unique(self.labels))
        print ("number of e_sigmas: %d number of labels: %d number of unique labels: %d\n" %(len(self.e_sigmas), len(self.labels), self.num_classes))

        # Split the given dataset to training and validation sets.
        self.num_train_samples = int(self.dsets.shape[1] * 0.9) * self.dsets.shape[0]
        self.num_valid_samples = int(self.dsets.shape[1] * 0.1) * self.dsets.shape[0]
        self.t_dsets = self.dsets[:,:int(self.dsets.shape[1] * 0.9), :]
        self.v_dsets = self.dsets[:,-int(self.dsets.shape[1] * 0.1):, :]

        # Preprocessing for Training
        #dsets = np.mean(self.t_dsets, axis=1)
        dsets = np.mean(self.dsets, axis=1)
        self.per_pixel_mean = np.array(dsets).astype(np.float32).mean(axis=0)
        self.per_pixel_mean = np.reshape(self.per_pixel_mean, (np.prod(self.per_pixel_mean.shape)))
        self.per_pixel_std = np.array(dsets).astype(np.float32).std(axis=0)
        self.per_pixel_std = np.reshape(self.per_pixel_std, (np.prod(self.per_pixel_std.shape)))

        dsets = np.reshape(self.t_dsets, (np.prod(self.t_dsets.shape[:-1]), self.t_dsets.shape[-1]))
        self.per_pixel_max = np.amax(dsets, axis=0)
        self.per_pixel_min = np.amin(dsets, axis=0)

        self.train_samples = np.reshape(self.t_dsets, (np.prod(self.t_dsets.shape[:-1]), self.t_dsets.shape[-1]))
        self.valid_samples = np.reshape(self.v_dsets, (np.prod(self.v_dsets.shape[:-1]), self.v_dsets.shape[-1]))

        self.train_labels = np.zeros((np.prod(self.t_dsets.shape[:-1]), self.num_classes))
        for i in range (np.prod(self.t_dsets.shape[:-1])):
            index = i // self.t_dsets.shape[1]
            label = self.labels[index]
            self.train_labels[i][label] = 1

        self.valid_labels = np.zeros((np.prod(self.v_dsets.shape[:-1]), self.num_classes))
        for i in range (np.prod(self.v_dsets.shape[:-1])):
            index = i // self.v_dsets.shape[1]
            label = self.labels[index]
            self.valid_labels[i][label] = 1

        #self.train_samples = (self.train_samples - self.per_pixel_min) / (self.per_pixel_max - self.per_pixel_min + 1e-10)
        #self.valid_samples = (self.valid_samples - self.per_pixel_min) / (self.per_pixel_max - self.per_pixel_min + 1e-10)
        #self.test_data = (self.test_data - self.per_pixel_min) / (self.per_pixel_max - self.per_pixel_min + 1e-10)

        self.train_samples = (self.train_samples - self.per_pixel_mean) / (self.per_pixel_std + 1e-10)
        self.valid_samples = (self.valid_samples - self.per_pixel_mean) / (self.per_pixel_std + 1e-10)
        self.test_data = (self.test_data - self.per_pixel_mean) / (self.per_pixel_std + 1e-10)

        self.num_train_batches = self.num_train_samples // (self.size * self.train_batch_size)
        self.num_local_train_samples = self.num_train_batches * self.train_batch_size
        self.local_train_samples_offset = self.rank * self.num_local_train_samples
        self.num_valid_batches = self.num_valid_samples // self.valid_batch_size

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

    def read_test_sample (self, sample_id):
        index = sample_id
        data = self.test_data[index]
        data = np.reshape(data, [self.input_length, 1])
        return data

    def tf_read_valid_sample (self, sample_id):
        data, label = tf.py_function(self.read_valid_sample, inp=[sample_id], Tout=[tf.float32, tf.float32])
        return data, label

    def tf_read_test_sample (self, sample_id):
        data = tf.py_function(self.read_test_sample, inp=[sample_id], Tout=[tf.float32])
        return data

    def valid_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_valid_samples))
        dataset = dataset.map(self.tf_read_valid_sample)
        dataset = dataset.batch(self.valid_batch_size)
        dataset = dataset.repeat()
        return dataset.__iter__()

    def test_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.test_data.shape[0]))
        dataset = dataset.map(self.tf_read_test_sample)
        dataset = dataset.batch(self.test_data.shape[0])
        dataset = dataset.repeat()
        return dataset.__iter__()

    def get_info (self, class_id):
        candidates = []
        for i in range (len(self.labels)):
            if self.labels[i] == class_id:
                candidates.append(i)
        e_sigmas = []
        e_cuts = []
        t_cuts = []
        for target in candidates:
            e_sigmas.append(self.e_sigmas[target])
            e_cuts.append(self.e_cuts[target])
            t_cuts.append(self.t_cuts[target])
        return candidates, e_sigmas, e_cuts, t_cuts
