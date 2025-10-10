import numpy as np
import tensorflow as tf
from mpi4py import MPI

def broadcast_model (model):
    num_params = len(model.trainable_variables)
    print ("Broadcasting %4d parameter sets from rank 0." % (num_params))
    for i in range (len(model.trainable_variables)):
        param = MPI.COMM_WORLD.bcast(model.trainable_variables[i], root = 0)
        model.trainable_variables[i].assign(param)

def average_grads (model, gradients):
    size = MPI.COMM_WORLD.Get_size()
    variables = model.trainable_variables
    global_grads = []
    if len(variables) == len(gradients):
        for i in range (len(variables)):
            param = model.trainable_variables[i]
            param_type = param.name.split("/")[1].split(":")[0]
            if param_type == "kernel" or param_type == "bias":
                global_grad = MPI.COMM_WORLD.allreduce(gradients[i], op = MPI.SUM) / size
                global_grads.append(global_grad)
            else:
                global_grads.append(gradients[i])
    else:
        for i in range (len(gradients)):
            global_grad = MPI.COMM_WORLD.allreduce(gradients[i], op = MPI.SUM) / size
            global_grads.append(global_grad)
    return global_grads
