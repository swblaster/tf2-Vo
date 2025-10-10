'''
Sunwoo Lee, Ph.D.
Postdoctoral Researcher
University of Southern California
<sunwool@usc.edu>
'''

import os
import csv
import numpy as np
import h5py

if __name__ == '__main__':
    f = h5py.File("data/data_1m.h5", 'r')
    data = f['PSD']
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    f.close()
    print (mean)
    print (std)

    f = open("mean_1m.txt", "w")
    for i in range (len(mean)):
        f.write("%f\n" % mean[i])
    f.close()

    f = open("std_1m.txt", "w")
    for i in range (len(std)):
        f.write("%f\n" % std[i])
    f.close()

    with open("mean_1m.txt", "r") as f:
        string_mean = f.readlines()
    mean = []
    for i in range(len(string_mean)):
        mean.append(float(string_mean[i]))
    mean = np.array(mean)
    print (mean)
    f.close()

    with open("std_1m.txt", "r") as f:
        string_std = f.readlines()
    std = []
    for i in range(len(string_std)):
        std.append(float(string_std[i]))
    std = np.array(std)
    print (std)
    f.close()
