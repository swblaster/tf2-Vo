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

class generator:
    def __init__ (self, num_traps, num_electrons, num_freqs):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_traps = num_traps
        self.num_electrons = num_electrons
        self.num_freqs = num_freqs
        self.max_depth = 150
        self.electron_mu = 0
        self.trap_sigma = 0.05

    def generate_electrons (self, e_sigma, e_cut):
        items = []
        length = self.num_electrons
        while len(items) < self.num_electrons:
            numbers = np.random.normal(self.electron_mu, e_sigma, length).astype(int)
            numbers = numbers[numbers > 0]
            numbers = numbers[numbers < e_cut]
            howmany = len(numbers)
            if len(items) + len(numbers) < self.num_electrons:
                length -= len(numbers)
                items += list(numbers)
            else:
                length = self.num_electrons - len(items)
                items += list(numbers[:length])
        self.E = np.array(items)
        self.E = self.comm.bcast(self.E, root = 0)
        bins = np.arange(self.max_depth + 1) # 0 ~ 150
        self.E_histo, self.E_bins = np.histogram(self.E, bins = bins)

    def generate_traps (self, trap_cut):
        self.T_histo = np.zeros((self.max_depth + 1)).astype(int)
        for i in range (trap_cut + 1, self.max_depth + 1):
            mu = 1 - 1/(i - trap_cut + 1) # alpha is 1.
            self.T_histo[i] = int(100 * abs(np.random.normal(mu, self.trap_sigma)))
        self.T_histo = self.comm.bcast(self.T_histo, root = 0)
        '''
        if self.rank == 0:
            for i in range (20):
                print ("Trap %3d: %d\n" %(trap_cut + i, self.T_histo[trap_cut + i]))
        '''

    def calculate_PSD(self, dset, index, e_sigma, e_cut):
        A = 1.0
        f_resolution = 32.0
        t0 = 11e-12
        lmd = 0.5
        S = np.zeros((self.num_freqs, e_cut + 1))

        for freq in range (1, 101):
            for elec in range (1, e_cut):
                for trap in range (1, self.max_depth):
                    exp_term = t0 * np.exp(abs(trap - elec) / lmd)
                    S[freq - 1][elec] += A * self.E_histo[elec] * self.T_histo[trap] * exp_term / (1 + (2 * np.pi * freq * f_resolution * exp_term)**2)
        self.PSD = np.sum(S, axis = 1)
        self.PSD = self.PSD * 1000 / self.PSD[0]
        dset[index] = self.PSD

if __name__ == '__main__':
    num_traps = 1000000
    num_electrons = 1000000
    num_samples_per_class = 100
    num_freqs = 100

    f = h5py.File('data_es1-14_ec7-20_tc4-24_alpha1.h5', 'w')
    e_sigmas = np.arange(1, 14) # 10 sigma values (1 ~ 13)
    e_cuts = np.arange(7, 20) # sigma + 6
    t_cuts = np.arange(4, 24) # 10 trap cut values (4 ~ 23)
    data = []
    class_id = 0
    for i in tqdm(range (len(e_sigmas))):
        e_sigma = e_sigmas[i]
        e_cut = e_cuts[i]
        for j in range (len(t_cuts)):
            t_cut = t_cuts[j]
            dset_name = "/" + str(class_id)
            print ("Creating sigma: %d cuts: %d dset_name: %s\n" %(e_sigma, t_cut, dset_name))
            if dset_name not in f:
                dset = f.create_dataset(dset_name, (num_samples_per_class, num_freqs,))
            else:
                dset = f.get(dset_name)
            for sample_id in range (num_samples_per_class):
                gen = generator(num_traps, num_electrons, num_freqs)
                gen.generate_electrons(e_sigma, e_cut)
                gen.generate_traps(t_cut)
                gen.calculate_PSD(dset, sample_id, e_sigma, e_cut)
            print ("class %3d\n" %(class_id))
            for i in range (len(dset[0])):
                print("%36.34f" %(dset[0][i]))
            print ("\n")
            class_id += 1
    f.close()
