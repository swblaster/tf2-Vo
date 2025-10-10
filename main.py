'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2023/03/17
Sunwoo Lee, Ph.D.
<sunwool@inha.ac.kr>
'''
import numpy as np
import tensorflow as tf
import config as cfg
from train import framework
from mpi4py import MPI
from solvers.sync_sgd import SyncSGD
from solvers.gsam import GSAM
from model import resnet20
from model import wideresnet28
from model import Network
from feeders.feeder_cifar import cifar
from feeders.feeder_vo import Vo

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_rank = rank % cfg.num_processes_per_node

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[local_rank], 'GPU')

    if cfg.dataset == "cifar10":
        batch_size = cfg.cifar10_config["batch_size"]
        num_epochs = cfg.cifar10_config["epochs"]
        min_lr = cfg.cifar10_config["min_lr"]
        max_lr = cfg.cifar10_config["max_lr"]
        num_classes = cfg.cifar10_config["num_classes"]
        decays = list(cfg.cifar10_config["decay"])
        weight_decay = cfg.cifar10_config["weight_decay"]

        dataset = cifar(batch_size = batch_size,
                        num_classes = num_classes)
    elif cfg.dataset == "cifar100":
        batch_size = cfg.cifar100_config["batch_size"]
        num_epochs = cfg.cifar100_config["epochs"]
        min_lr = cfg.cifar100_config["min_lr"]
        max_lr = cfg.cifar100_config["max_lr"]
        num_classes = cfg.cifar100_config["num_classes"]
        decays = list(cfg.cifar100_config["decay"])
        weight_decay = cfg.cifar100_config["weight_decay"]

        dataset = cifar(batch_size = batch_size,
                        num_classes = num_classes)
    elif cfg.dataset == "vo":
        batch_size = cfg.vo_config["batch_size"]
        num_epochs = cfg.vo_config["epochs"]
        min_lr = cfg.vo_config["min_lr"]
        max_lr = cfg.vo_config["max_lr"]
        decays = list(cfg.vo_config["decay"])
        input_length = cfg.vo_config["input_length"]
        weight_decay = cfg.vo_config["weight_decay"]
        cluster = cfg.vo_config["cluster"]

        dataset = Vo(batch_size = batch_size,
                     input_length = input_length, cluster = cluster)
        num_classes = dataset.num_classes
    else:
        print ("config.py has a wrong dataset definition.\n")
        exit()

    if rank == 0:
        print ("---------------------------------------------------")
        print ("dataset: " + cfg.dataset)
        print ("batch_size: " + str(batch_size))
        print ("training epochs: " + str(num_epochs))
        print ("---------------------------------------------------")

    if cfg.dataset == "cifar10":
        model = resnet20(weight_decay).build_model()
    elif cfg.dataset == "cifar100":
        model = wideresnet28(weight_decay).build_model()
    elif cfg.dataset == "vo":
        model = Network(input_length, weight_decay, num_classes).build_model()
    else:
        print ("Invalid dataset option!\n")
        exit()

    if cfg.optimizer == 0:
        solver = SyncSGD(num_classes = num_classes, model = model)
    elif cfg.optimizer == 1:
        solver = GSAM(num_classes = num_classes, model = model)
    else:
        print ("Invalid optimizer option!\n")
        exit()
    trainer = framework(model = model,
                        dataset = dataset,
                        solver = solver,
                        num_epochs = num_epochs,
                        min_lr = min_lr,
                        max_lr = max_lr,
                        decay_epochs = decays,
                        num_classes = num_classes,
                        do_checkpoint = cfg.checkpoint)
    trainer.train()
