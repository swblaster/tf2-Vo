'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2023/03/17
Sunwoo Lee, Ph.D.
<sunwool@inha.ac.kr>

Notice
 - "batch_size" is a local batch size per process.
 - The total batch size is the # of processes multiplied by the batch size.
'''

cifar10_config = {
    "batch_size": 64,
    "min_lr": 0.1,
    "max_lr": 0.1,
    "num_classes": 10,
    "epochs": 150,
    "decay": {100, 130},
    "weight_decay": 0.0001,
}

cifar100_config = {
    "batch_size": 64,
    "min_lr": 0.1,
    "max_lr": 0.1,
    "num_classes": 100,
    "epochs": 200,
    "decay": {150, 180},
    "weight_decay": 0.0005,
}

vo_config = {
    "batch_size": 128,
    "min_lr": 0.0001,
    "max_lr": 0.0001,
    "input_length": 100,
    "epochs": 100,
    "decay": {60, 80},
    "weight_decay": 0.0001,
    "cluster": -1,
}

num_processes_per_node = 8
dataset = "vo"
checkpoint = 1
'''
0: Synchronous SGD with data-parallelism
'''
optimizer = 0
