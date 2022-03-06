'''
All the settings are under an assumption of using 8 processes.
'''
oxygen_config = {
    "batch_size": 50,
    "min_lr": 0.1,
    "max_lr": 0.1,
    "num_classes": 10,
    "epochs": 200,
    "decay_epochs": [100, 170, 190],
    "input_length": 80,
    "input_path": "./data",
    "weight_decay": 0.0001,
}

num_procs_per_node = 4
dataset = "oxygen"
warmup_epochs = 0
checkpoint = 1
evaluate = 0
record = 0
cluster = 0
