'''
Sunwoo Lee, Ph.D.
Postdoctoral Researcher
University of Southern California
<sunwool@usc.edu>
'''

import argparse
from feeder import Feeder
from train import Trainer
from model import Network
import tensorflow as tf
from mpi4py import MPI
import config as cfg

if __name__ == '__main__':
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    local_rank = int(rank % cfg.num_procs_per_node) + 5
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[local_rank], 'GPU')

    batch_size = cfg.oxygen_config["batch_size"]
    min_lr = cfg.oxygen_config["min_lr"]
    max_lr = cfg.oxygen_config["max_lr"]
    num_classes = cfg.oxygen_config["num_classes"]
    num_epochs = cfg.oxygen_config["epochs"]
    decay_epochs = list(cfg.oxygen_config["decay_epochs"])
    input_length = cfg.oxygen_config["input_length"]
    input_path = cfg.oxygen_config["input_path"]
    weight_decay = cfg.oxygen_config["weight_decay"]
    do_evaluate = cfg.evaluate
    do_record = cfg.record
    do_checkpoint = cfg.checkpoint
    do_cluster = cfg.cluster

    network = Network(input_length)
    dataset = Feeder(input_path, batch_size, input_length, num_classes)
    trainer = Trainer(model = network,
                      dataset = dataset,
                      num_epochs = num_epochs,
                      min_lr = min_lr,
                      max_lr = max_lr,
                      weight_decay = weight_decay,
                      num_classes = num_classes,
                      decay_epochs = decay_epochs,
                      do_checkpoint = do_checkpoint)
    trainer.train()
    '''
    if do_evaluate == 1:
        trainer.evaluate()
    elif do_record == 1:
        trainer.record()
    elif do_cluster == 1:
        trainer.cluster()
    else:
        trainer.train()
    '''
