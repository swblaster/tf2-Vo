import random
import math
import numpy as np
from tqdm import tqdm
from mpi4py import MPI

class generator:
    def __init__ (self, num_traps, num_electrons):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_traps = num_traps
        self.num_electrons = num_electrons
        self.electron_mu = 0
        self.electron_sigma = 20
        self.negative_mu = 0
        self.negative_sigma = 40
        self.positive_mu = 0
        self.positive_sigma = 100

    def generate_electrons(self):
        items = []
        length = self.num_electrons
        while len(items) < self.num_electrons:
            numbers = np.random.normal(self.electron_mu, self.electron_sigma, length).astype(int)
            numbers = numbers[numbers>=0]
            howmany = len(numbers)
            if len(items) + len(numbers) < self.num_electrons:
                length -= len(numbers)
                items += list(numbers)
            else:
                length = self.num_electrons - len(items)
                items += list(numbers[:length])
        self.E = np.array(items)
        bins = np.arange(251)
        self.E_histo, self.E_bins = np.histogram(self.E, bins = bins)

    def generate_negative_distribution(self):
        items = []
        length = self.num_traps
        while len(items) < self.num_traps:
            numbers = np.random.normal(self.negative_mu, self.negative_sigma, length).astype(int)
            numbers = numbers[numbers>=0]
            howmany = len(numbers)
            if len(items) + howmany < self.num_traps:
                length -= howmany
            else:
                length = self.num_traps - len(items)
                numbers = numbers[:length]
            items += list(numbers)
        self.N = np.array(items)
        bins = np.arange(251)
        self.N_histo, self.N_bins = np.histogram(self.N, bins = bins)

    def generate_positive_distribution(self):
        items = []
        length = self.num_traps
        while len(items) < self.num_traps:
            numbers = np.random.normal(self.positive_mu, self.positive_sigma, length).astype(int)
            numbers = numbers[numbers>=0]
            howmany = len(numbers)
            if len(items) + howmany < self.num_traps:
                length -= howmany
            else:
                length = self.num_traps - len(items)
                numbers = numbers[:length]
            items += list(numbers)
        self.P = np.array(items)
        bins = np.arange(251)
        self.P_histo, self.P_bins = np.histogram(self.P, bins = bins)

    def calculate_trap_probability(self):
        # Calculate the probabilities.
        self.prob = np.array((self.P_histo - self.N_histo) / self.num_traps).astype(float)
        self.prob[self.prob<0] = 0

        # Normalize to be a sum of 1.
        self.prob = self.prob / sum(self.prob)

    def calculate_gate(self):
        self.gate = np.zeros((self.num_electrons, 250))
        for i in range (self.num_electrons):
            for j in range (250):
                if abs(j + 1 - self.E[i]) <= 50:
                    self.gate[i][j] = 1

    def calculate_PSD(self):
        A = 1.0
        f_resolution = 32.0
        t0 = 4e-15
        lmd = 0.5
        S = np.zeros((200, self.num_electrons))

        for i in tqdm(range (self.num_electrons)): # For each electron...
            for freq in range (200): # For each frequency...
                local_sum = 0
                for k in range (250): # For each 0.8 nm...
                    if self.gate[i][k] == 1:
                        K1 = t0 * np.exp(abs(k + 1 - self.E[i]) / lmd)
                        K2 = 1 + (2 * np.pi * (freq  + 1) * f_resolution * K1)**2
                        K = K1 / K2
                        local_sum += A * self.prob[k] * K
                S[freq][i] = local_sum
        self.PSD = np.sum(S, axis = 1)

if __name__ == '__main__':
    num_traps = 10000
    num_electrons = num_traps // 100

    gen = generator(num_traps, num_electrons)
    gen.generate_electrons()
    gen.generate_negative_distribution()
    gen.generate_positive_distribution()
    gen.calculate_trap_probability()
    gen.calculate_gate()
    gen.calculate_PSD()

    for i in range (len(gen.PSD)):
        print("%30.28f" %(gen.PSD[i]))
