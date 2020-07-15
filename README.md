# Objective

This little project contains studies on autoencoders applyed to anomaly detection.

All the scripts rely on the ```uad``` module developped in parallel available on github: https://github.com/hugovaysset/uad.

# Directories

* The directory ```draw_line_anomalies``` contains the examples of anomaly detection on MNIST with handmade "draw-line" anomalies. The directory contains studies on the decision functions used to detect anomalies from AE/VAE reconstructions, influence of the latent space dimension...
* The directory ```one_vs_all``` contains a more complete example of UAD on MNIST, by training a model on one digit (the "normal" class) and considering all the others to be abnormal. This experiment is a standard when it comes to evaluate a model on a UAD task. The directory contains several studies on different factors: architecture (influence of a self-attention layer, 4 contraction blocks instead of 3, latent space dimension, activation functions) and metrics.
performance of the VAE
* The directory ```ls_visualisation``` contains examples of visualisation techniques applied to inspect the latent space (t-SNE, PCA..)

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


