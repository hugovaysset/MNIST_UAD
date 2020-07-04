# Convolutional Variational Encoder Class

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, Input


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim1, dim2, dim3 = tf.shape(z_mean)[1], tf.shape(z_mean)[2], tf.shape(z_mean)[3]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim1, dim2, dim3))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):

    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        original_dim = 784
        intermediate_dim = 64
        k_size = 3
        dropout = 0.2
        batchnorm = False
        n_filters = 16
        latent_side = 4

        encoder_inputs = Input(shape=(28, 28, 1), name="encoder_inputs")

        paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0,
                                                         0]])  # shape d x 2 where d is the rank of the tensor and 2 represents "before" and "after"
        x = tf.pad(encoder_inputs, paddings, name="pad")

        # contracting path
        x = self.conv2d_block(x, n_filters * 1, kernel_size=k_size, batchnorm=batchnorm)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)

        x = self.conv2d_block(x, n_filters * 2, kernel_size=k_size, batchnorm=batchnorm)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)

        x = self.conv2d_block(x, n_filters=n_filters * 4, kernel_size=k_size, batchnorm=batchnorm)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)

        z_mean = layers.Conv2D(latent_dim, 1, strides=1, name="z_mean")(x)
        z_log_var = layers.Conv2D(latent_dim, 1, strides=1, name="z_log_var")(x)
        z = Sampling()((z_mean, z_log_var))

        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # Define decoder model.
        latent_inputs = Input(shape=(latent_side, latent_side, latent_dim), name="z_sampling")

        x = layers.Conv2DTranspose(n_filters * 4, (k_size, k_size), strides=(2, 2), padding='same', name="u6")(
            latent_inputs)
        x = layers.Dropout(dropout)(x)
        x = self.conv2d_block(x, n_filters * 4, kernel_size=k_size, batchnorm=batchnorm)

        x = layers.Conv2DTranspose(n_filters * 2, (k_size, k_size), strides=(2, 2), padding='same', name="u7")(x)
        x = layers.Dropout(dropout)(x)
        x = self.conv2d_block(x, n_filters * 2, kernel_size=k_size, batchnorm=batchnorm)

        x = layers.Conv2DTranspose(n_filters * 1, (k_size, k_size), strides=(2, 2), padding='same', name="u8")(x)
        x = layers.Dropout(dropout)(x)
        decoder_outputs = self.conv2d_block(x, 1, kernel_size=k_size, batchnorm=batchnorm)
        crop = tf.image.resize_with_crop_or_pad(decoder_outputs, 28, 28)

        self.decoder = Model(inputs=latent_inputs, outputs=crop, name="decoder")

    def conv2d_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=True):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        c1 = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                           kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            c1 = layers.BatchNormalization()(c1)
        c1 = layers.Activation('sigmoid')(c1)

        # second layer
        c1 = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                           kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            c1 = layers.BatchNormalization()(c1)
        c1 = layers.Activation('sigmoid')(c1)

        return c1

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
        return {"reconstruction_loss": reconstruction_loss}

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def generate_sample(self, n):
        """
        Generate a random sample in the latent space and returns the decoded images
        :param n: number of sample to generate
        :return:
        """
        latent_sample = np.array([tf.random.normal((n, self.latent_dim), mean=0.0, stddev=1.0)])
        latent_sample = np.array(tf.reshape(latent_sample, (n, *self.latent_dim)))
        generated = self.decoder.predict(latent_sample)
        return np.squeeze(generated, axis=-1)
