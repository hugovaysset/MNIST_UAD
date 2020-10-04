# Objective

This little project contains studies on autoencoders applyed to anomaly detection.

All the scripts rely on the ```uad``` module developped in parallel available on github: https://github.com/hugovaysset/uad.

# Directories

* The directory ```MNIST_draw_line_anomalies``` contains the examples of anomaly detection on MNIST with handmade "draw-line" anomalies. The directory contains studies on the decision functions used to detect anomalies from AE/VAE reconstructions, influence of the latent space dimension...
* The directory ```MNIST_one_vs_all``` contains a more complete example of UAD on MNIST, by training a model on one digit (the "normal" class) and considering all the others to be abnormal. This experiment is a standard when it comes to evaluate a model on a UAD task. The directory contains several studies on different factors: architecture (influence of a self-attention layer, 4 contraction blocks instead of 3, latent space dimension, activation functions) and metrics.
performance of the VAE
* The directory ```CIFAR_one_vs_all``` contains the same experiments done on CIFAR10 the dataset.
* The directory ```BACT_one_vs_all``` contains the same experiments on the real dataset of Mother Machine images.


