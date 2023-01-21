'''
Sunwoo Lee, Ph.D. <sunwool@inha.ac.kr>
Assistant Professor
Inha University, South Korea
Dec 23, 2022
'''

import numpy as np
import tqdm as tqdm
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
        self.sample_index = 0
        self.num_traps = num_traps
        self.A = 1.0
        self.f_resolution = 32
        self.trap_mu = 0
        self.trap_sigma = 50
        self.elec_mu = 0
        self.elec_sigma = 3
        self.lmd = lmd
        self.tau_0 = tau_0
        self.num_samples_per_sigma = num_samples_per_sigma
        self.unit_sigma = unit_sigma
        self.max_sigma = max_sigma
        self.unit_freq = unit_freq
        self.max_freq = max_freq

    def generate_traps (self):
        if self.trap_mu == 0 and self.trap_sigma == 0:
            print ("Set self.trap_sigma to a valid value, not zero!\n")
            exit()

        print ("1. Generating random %d trap locations...\n" %(self.num_traps))
        self.d = abs(np.random.normal(self.trap_mu, self.trap_sigma, self.num_traps)).astype(int)
        self.z = abs(np.random.normal(self.elec_mu, self.elec_sigma, self.num_traps)).astype(int)
        f = open("d.txt", "a")
        for i in range (len(self.d)):
            f.write("%f\n"%(self.d[i]))
        f.close()
        f = open("z.txt", "a")
        for i in range (len(self.z)):
            f.write("%f\n"%(self.z[i]))
        f.close()
        self.histo, bins = np.histogram(self.z, np.array(range(self.trap_sigma + 100)))
        histo_sum = sum(self.histo)
        print ("histo_sum = %d\n" %(histo_sum))
        f = open("h.txt", "a")
        for i in range (len(self.histo)):
            f.write("%d\n" %(self.histo[i]))
        f.close()

    def bound_range (self):
        print ("2. Bound the aggregation range...\n")
        self.gate = np.zeros((self.trap_sigma, self.num_traps))
        for i in range (self.num_samples_per_sigma):
            for j in range (self.num_traps):
                for neighbor in range (self.trap_sigma):
                    distance = abs(neighbor - self.d[j])
                    if distance <= 25:
                        self.gate[neighbor][j] = 1
                    else:
                        self.gate[neighbor][j] = 0

    def calc_noise (self):
        print ("3. Calculate the noise values...\n")
        self.S = np.zeros((self.max_freq, self.num_traps))
        t0 = 4e-15
        lmd = 0.5
        self.PSD = np.zeros((self.max_freq))

        '''
        constant = (2 * np.pi * self.f_resolution * t0)**2
        for freq in tqdm(range (1, self.max_freq + 1)):
            s = 0
            for z in range (self.trap_sigma):
                inner_s = 0
                if self.histo[z] == 0:
                    continue
                for trap_idx in range (self.num_traps):
                    exponent = abs(z - self.d[trap_idx]) / self.lmd
                    if exponent > 200:
                        exponent = 200
                    exp = np.exp(exponent)
                    exp2 = np.exp(2*exponent)
                    sub_s = (t0 * exp) / (1 + constant * freq**2 * exp2)
                    inner_s += self.A * self.histo[z] * self.gate[z][trap_idx] * sub_s
                s += inner_s
            self.PSD[freq - 1] = s
        '''

        # New code
        constant = (2 * np.pi * self.f_resolution * t0)**2
        for freq in tqdm(range (1, self.max_freq + 1)):
            s = 0
            for trap_idx in range (self.num_traps):
                inner_s = 0
                start = 0 if self.d[trap_idx] <= 25 else self.d[trap_idx] - 25
                end = self.d[trap_idx] + 25 if self.d[trap_idx] + 25 < self.trap_sigma + 100 else self.trap_sigma + 100 - 1
                #print ("start: %d d: %d end: %d\n" %(start, self.d[trap_idx], end))
                for z in range (start, end):
                    if self.histo[z] == 0:
                        continue
                    exponent = abs(z - self.d[trap_idx]) / self.lmd
                    if exponent > 200:
                        exponent = 200
                    exp = np.exp(exponent)
                    exp2 = np.exp(2*exponent)
                    sub_s = (t0 * exp) / (1 + constant * freq**2 * exp2)
                    if self.gate[z][trap_idx] != 1:
                        print ("trap_idx: %d z: %d\n" %(trap_idx, z))
                        exit()
                    #inner_s += self.A * self.histo[z] * self.gate[z][trap_idx] * sub_s
                    inner_s += self.A * self.histo[z] * sub_s
                s += inner_s
            self.PSD[freq - 1] = s

if __name__ == '__main__':
    gen = generator(cfg.num_traps, cfg.lmd, cfg.tau_0, cfg.num_samples_per_sigma,
                    cfg.unit_sigma, cfg.max_sigma, cfg.unit_freq, cfg.max_freq)
    gen.trap_sigma = 25
    gen.trap_mu = 5
    gen.elec_sigma = 3
    gen.elec_mu = 0
    gen.sample_index = 0
    gen.generate_traps()
    gen.bound_range()
    gen.calc_noise()

    name = "PSD_t" + str(gen.trap_sigma) + "_e" + str(gen.sample_index) + ".txt"
    f = open(name, "a")
    for i in range (len(gen.PSD)):
        f.write("%f\n" %(gen.PSD[i]))
    f.close()

    if gen.rank == 0:
        print ("All done.\n")
