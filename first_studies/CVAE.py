# Convolutional Variational Encoder Class

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

original_dim = 784
intermediate_dim = 64
latent_dim = 16
latent_side = int(np.sqrt(latent_dim))
kernel_size = 3
dropout = 0
batchnorm = False
n_filters = 4


# NE PAS UTILISER keras MAIS tf.keras

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim1, dim2, dim3 = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim1, dim2, dim3))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    def __init__(self, encoder, decoder, dims=(28, 28, 1), latent_dim=2, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.dims = dims
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

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
        latent_sample = np.array(tf.reshape(latent_sample, (n, self.latent_dim)))
        generated = self.decoder.predict(latent_sample)
        return np.squeeze(generated, axis=-1)


if __name__ == '__main__':
    original_dim = 784
    latent_dim = 2
    latent_side = int(np.sqrt(latent_dim))
    kernel_size = 3
    dropout = 0
    batchnorm = False
    n_filters = 4

    encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, input_shape=(28, 28, 1), activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # encoder.summary()

    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(encoder, decoder, dims=(28, 28, 1), latent_dim=2)

    vae.compile(optimizer=tf.keras.optimizers.Adam())

    pred = vae.generate_sample(4)

    print(pred.shape)



