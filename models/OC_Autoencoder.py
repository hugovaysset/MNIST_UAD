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


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim1, dim2, dim3 = tf.shape(z_mean)[1], tf.shape(z_mean)[2], tf.shape(z_mean)[3]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim1, dim2, dim3))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class SVDD_VAE(Model):
    def __init__(self, encoder, decoder, dims=(28, 28, 1), latent_dim=2, LAMBDA=1e-6, **kwargs):
        super(SVDD_VAE, self).__init__(**kwargs)
        self.dims = dims
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

        self.CENTER = 1  # find a way to get mean value after the first forward pass
        self.LAMBDA = LAMBDA

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(data, reconstruction))
            reconstruction_loss *= 28 * 28

            size = tf.shape(z_mean)[1] + tf.shape(z_mean)[2] + tf.shape(z_mean)[3]
            distance_loss = (1 / size) * tf.cast(tf.reduce_sum((z - self.CENTER) ** 2), dtype=tf.double)
            weight_decay = 0
            for lay in self.encoder.layers:
                if lay.trainable_weights != []:
                    # first norm: compute the Frobenius norm of each kernel -> (n_feature_maps_input, n_fm_output)
                    # second norm: compute the Frobenius norm on the remaining matrix
                    weight_decay += tf.cast(tf.norm(tf.norm(lay.trainable_weights[0], axis=(-2, -1), ord="fro") ** 2,
                                         axis=(-2, -1), ord="fro"), dtype=tf.float64)
            for lay in self.decoder.layers:
                if lay.trainable_weights != []:
                    weight_decay += tf.cast(tf.norm(tf.norm(lay.trainable_weights[0], axis=(-2, -1), ord="fro") ** 2,
                                         axis=(-2, -1), ord="fro"), dtype=tf.float64)
            weight_decay *= self.LAMBDA / 2

            svdd_loss = weight_decay + distance_loss

            total_loss = reconstruction_loss + tf.cast(svdd_loss, dtype=tf.float32)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "svdd_loss": svdd_loss,
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


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, activation1=True, activation2=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
                      kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation1:
        x = layers.Activation('sigmoid')(x)

    # second layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
                      kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation2:
        x = layers.Activation('sigmoid')(x)

    return x


if __name__ == '__main__':
    original_dim = 784
    intermediate_dim = 64
    kernel_size = 3
    dropout = 0.2
    batchnorm = False
    n_filters = 16
    latent_dim = n_filters * 1
    latent_side = 4

    # Define encoder model.
    encoder_inputs = tf.keras.Input(shape=(28, 28, 1), name="encoder_inputs")

    paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0,
                                                     0]])  # shape d x 2 where d is the rank of the tensor and 2 represents "before" and "after"
    x = tf.pad(encoder_inputs, paddings, name="pad")

    # contracting path
    x = conv2d_block(x, n_filters * 1, kernel_size=kernel_size, batchnorm=batchnorm)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout)(x)

    x = conv2d_block(x, n_filters * 2, kernel_size=kernel_size, batchnorm=batchnorm)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout)(x)

    x = conv2d_block(x, n_filters=n_filters * 4, kernel_size=kernel_size, batchnorm=batchnorm)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout)(x)

    z_mean = layers.Conv2D(latent_dim, 1, strides=1, name="z_mean")(x)
    z_log_var = layers.Conv2D(latent_dim, 1, strides=1, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))

    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Define decoder model.
    latent_inputs = tf.keras.Input(shape=(latent_side, latent_side, latent_dim), name="z_sampling")

    x = layers.Conv2DTranspose(n_filters * 4, (kernel_size, kernel_size), input_shape=(latent_side, latent_side, 1),
                               strides=(2, 2), padding='same', name="u6")(latent_inputs)
    x = layers.Dropout(dropout)(x)
    x = conv2d_block(x, n_filters * 4, kernel_size=kernel_size, batchnorm=batchnorm)

    x = layers.Conv2DTranspose(n_filters * 2, (kernel_size, kernel_size), strides=(2, 2), padding='same', name="u7")(x)
    x = layers.Dropout(dropout)(x)
    x = conv2d_block(x, n_filters * 2, kernel_size=kernel_size, batchnorm=batchnorm)

    x = layers.Conv2DTranspose(n_filters * 1, (kernel_size, kernel_size), strides=(2, 2), padding='same', name="u8")(x)
    x = layers.Dropout(dropout)(x)
    decoder_outputs = conv2d_block(x, 1, kernel_size=kernel_size, batchnorm=batchnorm, activation1=True,
                                   activation2=True)
    crop = tf.image.resize_with_crop_or_pad(decoder_outputs, 28, 28)

    decoder = tf.keras.Model(inputs=latent_inputs, outputs=crop, name="decoder")

    vae = SVDD_VAE(encoder, decoder, LAMBDA=1e-6)

    from tensorflow.keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = np.expand_dims(x_train, axis=-1).astype('float64') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float64') / 255.0

    vae.compile(optimizer=tf.keras.optimizers.Adam())
    vae.fit(x_train, epochs=5, batch_size=128)
