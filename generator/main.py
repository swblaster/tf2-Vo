'''
Sunwoo Lee, Ph.D. <sunwool@inha.ac.kr>
Assistant Professor
Inha University, South Korea
Dec 23, 2022
'''

import numpy as np
import math
import time
from tqdm import tqdm
from mpi4py import MPI
import config as cfg

class generator:
    def __init__ (self, num_traps, lmd, tau_0, num_samples_per_sigma,
                  unit_sigma, max_sigma, unit_freq, max_freq):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_traps = num_traps
        self.A = 1.0
        self.f_resolution = 32
        self.mu = 0
        self.sigma = 0
        self.lmd = lmd
        self.tau_0 = tau_0
        self.num_samples_per_sigma = num_samples_per_sigma
        self.unit_sigma = unit_sigma
        self.max_sigma = max_sigma
        self.unit_freq = unit_freq
        self.max_freq = max_freq

    def generate_traps (self):
        if self.mu == 0 and self.sigma == 0:
            print ("Set self.sigma to a valid value, not zero!\n")
            exit()

        print ("1. Generating random %d trap locations...\n" %(self.num_traps))
        self.d = abs(np.random.normal(self.mu, self.sigma, self.num_traps))
        self.histo, bins = np.histogram(self.d, np.array(range(self.sigma + 100)))
        histo_sum = sum(self.histo)
        print ("histo_sum = %d\n" %(histo_sum))

    def bound_range (self):
        print ("2. Bound the aggregation range...\n")
        self.gate = np.zeros((self.sigma, self.num_traps))
        for i in range (self.num_samples_per_sigma):
            for j in range (self.num_traps):
                for neighbor in range (self.sigma):
                    distance = abs(neighbor - self.z[j])
                    if distance <= 50:
                        self.gate[neighbor][j] = 1
                    else:
                        self.gate[neighbor][j] = 0

    def calc_noise (self):
        print ("3. Calculate the noise values...\n")
        self.S = np.zeros((self.max_freq, self.num_traps))
        t0 = 4e-15
        lmd = 0.5

        for trap_idx in range (self.num_traps):
            for freq in range (1, self.max_freq):
                s = 0
                for z in range (self.sigma):
                    exp = np.exp(abs(z - self.d[trap_idx]) / self.lmd)
                    sub_A = ((t0 * exp ) / (1 + (2 * np.pi * freq * self.f_resolution * (t0 * exp))**2))
                    s += self.A * self.histo(z) * self.gate(z) * sub_A
                self.S[freq][trap_idx] = s

    def sum_up (self):
        print ("4. Sum up the noise values to create PSD curves...\n")

if __name__ == '__main__':
    gen = generator(cfg.num_traps, cfg.lmd, cfg.tau_0, cfg.num_samples_per_sigma,
                    cfg.unit_sigma, cfg.max_sigma, cfg.unit_freq, cfg.max_freq)
    gen.sigma = 25
    gen.generate_traps()
    gen.bound_range()
    gen.calc_noise()
    gen.sum_up()

    if gen.rank == 0:
        print ("All done.\n")
