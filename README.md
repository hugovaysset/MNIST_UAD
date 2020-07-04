# Objective

This little project contains the first studies on autoencoders applyed to anomaly detection.

# Directory

* The directory ```first_studies``` contains the examples of autoencoders and variational autoencoders applied
to anomaly detection on MNIST digits
* The directory ```ls_dimension``` contains a small study on the impact of the latent space dimension on the 
performance of the VAE
* The directory ```ls_prior``` contains a study on the impact of choosing a gaussian mixture as a prior distrib
instead of a simple gaussian distribution (to be done).

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


