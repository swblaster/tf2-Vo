# Polar-Component Estimator
This repository contains the software framework for experiments conducted in a preprint of our paper, `Complementary Electro-Ionic Reconstruction Probed by Machine Learning-assisted Noise Spectroscopy`.

## Software Requirements
1. TensorFlow >= 2.15.0
2. Python3
3. HDF5 

## Feature Finder
`cluster.py` receives a simulated PSD curve dataset stored in an HDF5 file and cluster them using DBSCAN algorithm. Each pair of the standard deviation and the onset position of oxygen vacancies is assigned with a cluster ID.

## Data Generator
`gen.py` calculates PSD curves and prints out them.

## Deep Learning Framework
`main.py` classifies the given 'observed' PSD curves based on the clusters found by our Feature Finder. The classifier is a manually designed 1-D Convolutional Neural Network. This code is based on the basic DL framework from LMLS lab at Inha University. The details will be updated soon.

## Contacts
 * Sunwoo Lee <sunwool@inha.ac.kr>
