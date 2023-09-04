import numpy as np
from keras import layers, models
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
import tensorflow_addons as tfa


class SimpleGAN:
    def __init__(self, img_size, latent_dim=100):
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.channels = 3
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.gan = self.build_gan()

    def build_discriminator(self):
        img_input = layers.Input(shape=(self.img_size, self.img_size, self.channels))

        x = tfa.layers.SpectralNormalization(layers.Conv2D(64, kernel_size=3, strides=2, padding='same'))(img_input)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = tfa.layers.SpectralNormalization(layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = tfa.layers.SpectralNormalization(layers.Conv2D(256, kernel_size=3, strides=2, padding='same'))(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Flatten()(x)
        output = layers.Dense(1)(x)

        model = Model(img_input, output)
        return model

    def build_generator(self):
        noise_input = layers.Input(shape=(self.latent_dim,))

        x = layers.Dense(8 * 8 * 256)(noise_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Reshape((8, 8, 256))(x)

        x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2DTranspose(self.channels, kernel_size=3, strides=2, padding='same')(x)
        output = layers.Activation('tanh')(x)

        model = Model(noise_input, output)
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        noise_input = layers.Input(shape=(self.latent_dim,))
        gen_img = self.generator(noise_input)
        validity = self.discriminator(gen_img)

        model = Model(noise_input, validity)
        return model

    def gradient_penalty(self, real_img, fake_img, batch_size):
        epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        interpolated_img = epsilon * real_img + (1 - epsilon) * fake_img

        with tf.GradientTape() as tape:
            tape.watch(interpolated_img)
            validity = self.discriminator(interpolated_img)

        gradients = tape.gradient(validity, interpolated_img)
        gradients_norm = tf.norm(tf.reshape(gradients, [batch_size, -1]), axis=1)
        gp = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        return gp

    def train(self, x_train, epochs, batch_size=128, n_critic=5):
        optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

        for epoch in range(epochs):
            for _ in range(n_critic):
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                real_imgs = x_train[idx]

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                with tf.GradientTape() as tape:
                    real_validity = self.discriminator(real_imgs)
                    fake_validity = self.discriminator(gen_imgs)
                    gp = self.gradient_penalty(real_imgs, gen_imgs, batch_size)

                    d_loss = tf.reduce_mean(fake_validity - real_validity + 10 * gp)

                d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
                optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                gen_imgs = self.generator(noise)
                fake_validity = self.discriminator(gen_imgs)
                g_loss = -tf.reduce_mean(fake_validity)

            g_grads = tape.gradient(g_loss, self.generator.trainable_weights)
            optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

            if epoch % 100 == 0:
                print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
