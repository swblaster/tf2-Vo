'''
All the settings are under an assumption of using 8 processes.
'''
oxygen_config = {
    "batch_size": 10,
    "min_lr": 0.1,
    "max_lr": 0.1,
    "num_classes": 100,
    "epochs": 50,
    "decay_epochs": {25, 40},
    "input_length": 80,
    "input_path": "./data",
    "weight_decay": 0.0001,
}

num_procs_per_node = 4
dataset = "oxygen"
warmup_epochs = 0
checkpoint = 0
evaluate = 0
record = 0
cluster = 0
