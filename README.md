# Polar-Component Estimator
This repository contains the software framework for experiments conducted in a preprint of our paper, `Complementary Electro-Ionic Reconstruction Probed by Machine Learning-assisted Noise Spectroscopy`.

## Software Requirements
1. TensorFlow >= 2.15.0
2. Python3
3. HDF5
4. MPI

## Feature Finder
`cluster.py` receives a simulated PSD curve dataset stored in an HDF5 file and cluster them using DBSCAN algorithm. Each pair of the standard deviation and the onset position of oxygen vacancies is assigned with a cluster ID.
 - Electron Sigma Values (rows): 1~11
 - Onset Position of Traps (columns): 1~23
 - `python cluster.py` reads the input HDF file, cluster them using DBSCAN, and generate the result as a png file as shown below.

![map](https://github.com/swblaster/tf2-Vo/blob/master/map.png)

## Simulation Data Generator
 - `gen.py` calculates PSD curves and prints out them.
 - The generated simulation data will be stored as a single HDF5 file. All samples corresponding to an individual setting will be stored as a single dataset under the root group.

## Classifier (Deep Learning Framework)
`main.py` classifies the given 'observed' PSD curves based on the clusters found by our Feature Finder. The classifier is a manually designed 1-D Convolutional Neural Network. This code is based on the basic DL framework from LMLS lab at Inha University. Please see the paper's supplementary document for detailed hyper-parameter settings used to train the neural network on simulation data.
 - `python main.py` evaluates the test samples stored in `observation.bin` using the neural network model stored in `checkpoint_0`.
 - To train a model from the scratch, delete `checkpoint_0` and run `python main.py`. Then, the program will begin the training and record the model at the end of every epoch.

## Extra Features
- The Classifier program has been parallelized using MPI. If you have multiple GPUs on your system, you can train the neural network `mpirun -n 2 pyton main`. Then, the training will be performed on 2 GPUs.

## Contacts
 * Sunwoo Lee <sunwool@inha.ac.kr>
