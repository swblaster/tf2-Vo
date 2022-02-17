import numpy as np
import math
import time
from tqdm import tqdm
from mpi4py import MPI

mu = 1.0
sigma = 400
N = 1000 # Number of data for each sigma
T = 12500 # Number of traps
t0 = 1e-9
lambd = 1.78e-8
A = 1.0
f_resolution = 128
L = 200 # Number of output points per sample

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Step 1
for k in tqdm(range (3)):
    t1 = time.time()
    np.random.seed(rank + int(time.time()))
    z = np.random.normal(mu, sigma, N*T)
    z = np.reshape(z, (N, T))
    for i in range (N):
        for j in range (T):
            if abs(z[i,j]) > 500:
                z[i,j] = 0
                #while abs(z[i,j]) > 500:
                #    z[i,j] = np.random.normal(mu, sigma, 1)
    z = z * 2e-8
    z = abs(z)
    print ("Step1: %f sec\n" %(time.time() - t1))

    # Step 2
    t1 = time.time()
    t = np.zeros((N, T))
    for i in range (N):
        for j in range (T):
            if z[i,j] != 0:
                t[i,j] = t0 * math.exp(z[i,j] / lambd)
            else:
                t[i,j] = 0.0
    print ("Step2: %f sec\n" %(time.time() - t1))

    # Step 3
    t1 = time.time()
    s = np.zeros((N, L))
    for i in range (N):
        for f in range (1, L + 1):
            single_s = 0
            for j in range (T):
                tt = t[i,j]
                single_s += A * tt / (1 + (2 * math.pi * f * f_resolution)**2 * tt**2)
            s[i,f - 1] = single_s
    
    comm.Barrier()
    global_s = comm.allreduce(s, op = MPI.SUM)
    print ("Step3: %f sec\n" %(time.time() - t1))

    # Step 4
    t1 = time.time()
    name = "sigma_" + str(sigma) + ".txt"
    f = open(name, "w")
    for row in global_s:
        np.savetxt(f, row)
    f.close()
    print ("Step4: %f sec\n" %(time.time() - t1))

    sigma += 50
