import random
import math
import numpy as np
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
        self.electron_mu = 0
        self.electron_sigma = 10
        self.trap_mu = 120
        self.trap_sigma = 20
        self.electron_cut = 35
        self.trap_cut = 40

    def generate_electrons(self):
        items = []
        length = self.num_electrons
        while len(items) < self.num_electrons:
            numbers = np.random.normal(self.electron_mu, self.electron_sigma, length).astype(int)
            numbers = numbers[numbers >= 0]
            numbers = numbers[numbers < self.electron_cut]
            howmany = len(numbers)
            if len(items) + len(numbers) < self.num_electrons:
                length -= len(numbers)
                items += list(numbers)
            else:
                length = self.num_electrons - len(items)
                items += list(numbers[:length])
        self.E = np.array(items)
        self.E = self.comm.bcast(self.E, root = 0)
        bins = np.arange(151)

        # Replace one random sample with the deepest electron.
        #self.E[0] = self.electron_cut + 1
        self.E_histo, self.E_bins = np.histogram(self.E, bins = bins)
        if self.rank == 0:
            for i in range (10):
                offset = self.electron_cut - i
                print ("Elec %3d: %d\n" %(self.E_bins[offset], self.E_histo[offset]))

    def generate_positive_distribution(self):
        items = []
        length = self.num_traps
        while len(items) < self.num_traps:
            numbers = np.random.normal(self.trap_mu, self.trap_sigma, length).astype(int)
            numbers = numbers[numbers > self.trap_cut]
            #numbers = numbers[numbers < 150]
            howmany = len(numbers)
            if len(items) + howmany < self.num_traps:
                length -= howmany
            else:
                length = self.num_traps - len(items)
                numbers = numbers[:length]
            items += list(numbers)
        self.P = np.array(items)
        #self.P[0] = self.trap_cut - 1
        self.P = self.comm.bcast(self.P, root = 0)
        bins = np.arange(151)
        self.P_histo, self.P_bins = np.histogram(self.P, bins = bins)
        for i in range (10):
            offset = self.trap_cut + i
            self.P_histo[offset] = max(0, i-6)
            self.P_histo[offset] *= 10
            if self.rank == 0:
                print ("Trap %3d: %d\n" %(self.P_bins[offset], self.P_histo[offset]))

    def calculate_trap_probability(self):
        self.prob = np.array(self.P_histo).astype(float)
        self.prob = self.prob / sum(self.prob)

    def calculate_gate(self):
        self.gate = np.zeros((self.num_electrons, 150))
        for i in range (self.num_electrons):
            for j in range (150):
                if abs(j + 1 - self.E[i]) <= 50:
                    self.gate[i][j] = 1

    def calculate_PSD(self):
        A = 1.0
        f_resolution = 32.0
        t0 = 4e-15
        lmd = 0.5
        S = np.zeros((self.num_freqs, self.num_electrons))

        length = self.num_electrons // self.size
        remainder = self.num_electrons % self.size
        if remainder != 0:
            print ("The number of electrons is not divisible by the number of processes\n")
            exit()
        offset = self.rank * length
        #for i in tqdm(range (self.num_electrons)): # For each electron...
        for i in tqdm(range (offset, offset + length)): # For each electron...
            for freq in range (self.num_freqs): # For each frequency...
                local_sum = 0
                for k in range (150): # For each 0.8 nm...
                    if self.gate[i][k] == 1:
                        K1 = t0 * np.exp(abs(k + 1 - self.E[i]) / lmd)
                        K2 = 1 + (2 * np.pi * (freq  + 1) * f_resolution * K1)**2
                        K = K1 / K2
                        local_sum += A * self.prob[k] * K
                S[freq][i] = local_sum
        self.local_PSD = np.sum(S, axis = 1)
        self.PSD = self.comm.allreduce(self.local_PSD, op = MPI.SUM)

if __name__ == '__main__':
    num_traps = 100000
    num_electrons = 100000
    num_freqs = 100

    gen = generator(num_traps, num_electrons, num_freqs)
    gen.generate_electrons()
    gen.generate_positive_distribution()
    gen.calculate_trap_probability()
    gen.calculate_gate()
    gen.calculate_PSD()

    if gen.rank == 0:
        for i in range (len(gen.PSD)):
            print("%36.34f" %(gen.PSD[i]))
