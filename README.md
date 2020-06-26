# Objective

This little project contains the first studies on autoencoders applyed to anomaly detection.

# Directory

* The notebook ```MNIST_UAD.ipynb``` contains the main study.
* The ```MNIST_anomalies``` module contains all the utility functions to generate and plot the anomalies
generated on the images.
* The ```CVAE.py``` module contains two classes that build the convolutional variational autoencoder used
in the last part of the study.

# Material and methods

As a first example, we chose to work on the MNIST dataset on which we generate simple anomalies which are
straight lines at a random location in the image. Several autoencoders architectures are trained on the 
training set of MNIST (60,000 images) and evaluated on the test set (10,000 images). The tested architectures
are :
* Simple dense autoencoder
* Convolutional autoencoder
* Dense VAE
* Convolutional VAE using either a U-Net based architecture or a simpler architecture with two convolutions
in the encoder et two transpose-convolutions in the decoder.

During training, the loss is the sum of a binary cross entropy and KL divergence.

# Results
The CVAE with the simple architecture seems to perform the best at detecting and segmenting the anomalies.


