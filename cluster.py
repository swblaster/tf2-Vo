'''
Sunwoo Lee, Ph.D.
Assistant Professor
Inha University, Republic of Korea
<sunwool@inha.ac.kr>
2025/10/16
'''
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
    max_e_sigma = 10

    sigmas = []
    cuts = []
    for i in range (len(e_sigmas)):
        sigma = e_sigmas[i]
        if i > max_e_sigma:
            break
        for j in range (len(t_cuts)):
            cut = min_cut + j
            sigmas.append(sigma)
            cuts.append(cut)
            class_id = i*len(t_cuts) + j
            dset = np.array(f["/" + str(class_id)])
            dsets.append(dset)
            #if sigma == 4 and cut == 12:
            #    for k in range (100):
            #        print ("%f" %(dset[0][k]))
    dsets = np.array(dsets)
    dimensions = dsets.shape
    num_blocks = i * len(t_cuts)

    # The data should be transposed before making label.
    # Otherwise, the left bottom classes are assigned with a large class ID, making the colormap unnatural...
    dsets = np.reshape(dsets, (max_e_sigma + 1, len(t_cuts), 100, 100))
    #dsets = np.transpose(dsets, (1, 0, 2, 3))
    dsets = np.reshape(dsets, (num_blocks, 100, 100))
    dsets = np.mean(dsets, axis=1)
    clustering = DBSCAN(eps=2.0, min_samples=1).fit(dsets)
    #labels = np.reshape(clustering.labels_, ((len(t_cuts), max_e_sigma + 1)))
    labels = np.reshape(clustering.labels_, ((max_e_sigma + 1, len(t_cuts))))
    #labels = np.transpose(labels, (1, 0))
    labels = clustering.labels_.flatten()
    f = open("cluster.txt", "w")
    for i in range (len(labels)):
        f.write("%d\n" %(labels[i]))
    f.close()
    for i in range (max_e_sigma + 1):
        index = i * len(t_cuts)
        print (labels[index:index + len(t_cuts)])
