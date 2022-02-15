'''
All the settings are under an assumption of using 8 processes.
'''
oxygen_config = {
    "batch_size": 50,
    "min_lr": 0.001,
    "max_lr": 0.001,
    "num_classes": 100,
    "epochs": 100,
    "decay_epochs": {75, 90},
    "input_length": 200,
    "latent_length": 32,
    "input_path": "./data",
    "weight_decay": 0.0001,
}

num_procs_per_node = 4
dataset = "oxygen"
warmup_epochs = 0
checkpoint = 0
evaluate = 0
record = 1
cluster = 0
