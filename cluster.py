import random
import math
import numpy as np
import h5py
from tqdm import tqdm
from mpi4py import MPI
from sklearn.cluster import DBSCAN

if __name__ == '__main__':
    # Read all the files.
    f = h5py.File("data_es1-16_ec7-22_tc4-24_alpha1.h5", 'r')
    dsets = []
    min_sigma = 1
    min_cut = 4
    e_sigmas = np.arange(1, 16)
    e_cuts = np.arange(7, 22)
    t_cuts = np.arange(4, 24)
    num_e_sigmas = len(e_sigmas)
    num_cuts = len(t_cuts)

    sigmas = []
    cuts = []
    for i in range (num_e_sigmas):
        sigma = e_sigmas[i]
        if i > 11:
            continue
        for j in range (num_cuts):
            cut = min_cut + j
            sigmas.append(sigma)
            cuts.append(cut)
            class_id = i*num_cuts + j
            dset = np.array(f["/" + str(class_id)])
            dsets.append(dset)
            if sigma == 4 and cut == 12:
                for k in range (100):
                    print ("%f" %(dset[0][k]))
    dsets = np.array(dsets)
    # The data should be transposed before making label.
    # Otherwise, the left bottom classes are assigned with a large class ID, making the colormap unnatural...
    dsets = np.reshape(dsets, (15, 20, 100, 100))
    dsets = np.transpose(dsets, (1, 0, 2, 3))
    dsets = np.reshape(dsets, (300, 100, 100))
    dsets = np.mean(dsets, axis=1)
    clustering = DBSCAN(eps=2.0, min_samples=1).fit(dsets)
    labels = np.reshape(clustering.labels_, ((num_cuts, num_e_sigmas)))
    labels = np.transpose(labels, (1, 0))
    labels = clustering.labels_.flatten()
    f = open("label.txt", "w")
    for i in range (len(labels)):
        f.write("%d\n" %(labels[i]))
    f.close()
    for i in range (len(e_sigmas)):
        index = i * len(t_cuts)
        print (labels[index:index + len(t_cuts)])
