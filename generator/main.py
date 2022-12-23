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
        self.d = np.random.normal(self.mu, self.sigma, self.num_traps)
        for i in range (len(self.d)):
            print ("%d\n" %(self.d[i]))

    def bound_range (self):
        print ("2. Bound the aggregation range...\n")

    def calc_noise (self):
        print ("3. Calculate the noise values...\n")

    def sum_up (self):
        print ("4. Sum up the noise values to create PSD curves...\n")

if __name__ == '__main__':
    gen = generator(cfg.num_traps, cfg.lmd, cfg.tau_0, cfg.num_samples_per_sigma,
                    cfg.unit_sigma, cfg.max_sigma, cfg.unit_freq, cfg.max_freq)
    gen.generate_traps()
    gen.bound_range()
    gen.calc_noise()
    gen.sum_up()

    if gen.rank == 0:
        print ("All done.\n")
