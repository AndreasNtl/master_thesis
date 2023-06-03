import numpy as np
from keras import layers, Sequential
from keras.layers import Dense, Reshape, Conv2DTranspose, ReLU, Conv2D, LeakyReLU, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf


class SimpleGAN:
    def __init__(self, img_size, latent_dim=100):
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.channels = 3
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.gan = self.build_gan()

    def wasserstein_loss(self, y_true, y_pred):
        return tf.keras.backend.mean(y_true * y_pred)

    def build_discriminator(self, in_shape=(64, 64, 3)):
        model = Sequential(name="Discriminator")  # Model

        # Hidden Layer 1
        model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape,
                         name='Discriminator-Hidden-Layer-1'))
        model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-1'))

        # Hidden Layer 2
        model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape,
                         name='Discriminator-Hidden-Layer-2'))
        model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-2'))

        # Hidden Layer 3
        model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape,
                         name='Discriminator-Hidden-Layer-3'))
        model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-3'))

        # Flatten and Output Layers
        model.add(Flatten(name='Discriminator-Flatten-Layer'))  # Flatten the shape
        # Randomly drop some connections for better generalization
        model.add(Dropout(0.3,
                          name='Discriminator-Flatten-Layer-Dropout'))
        model.add(Dense(1, activation='sigmoid', name='Discriminator-Output-Layer'))  # Output Layer

        # Compile the model
        #model.compile(loss=self.wasserstein_loss, optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        #              metrics=['accuracy'])

        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                      metrics=['accuracy'])
        return model

    def build_generator(self):
        model = Sequential(name="Generator")  # Model

        # Hidden Layer 1: Start with 8 x 8 image
        n_nodes = 8 * 8 * 128  # number of nodes in the first hidden layer
        model.add(Dense(n_nodes, input_dim=self.latent_dim, name='Generator-Hidden-Layer-1'))
        model.add(Reshape((8, 8, 128), name='Generator-Hidden-Layer-Reshape-1'))

        # Hidden Layer 2: Upsample to 16 x 16
        model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                  name='Generator-Hidden-Layer-2'))
        model.add(ReLU(name='Generator-Hidden-Layer-Activation-2'))

        # Hidden Layer 3: Upsample to 32 x 32
        model.add(Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                  name='Generator-Hidden-Layer-3'))
        model.add(ReLU(name='Generator-Hidden-Layer-Activation-3'))

        # Hidden Layer 4: Upsample to 64 x 64
        model.add(Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                  name='Generator-Hidden-Layer-4'))
        model.add(ReLU(name='Generator-Hidden-Layer-Activation-4'))

        # Output Layer (Note, we use 3 filters because we have 3 channels for a color image. Grayscale would have
        # only 1 channel)
        model.add(
            Conv2D(filters=3, kernel_size=(5, 5), activation='tanh', padding='same', name='Generator-Output-Layer'))
        return model

    # def build_discriminator(self, in_shape=(256, 256, 3)):
    #     model = Sequential(name="Discriminator")  # Model
    #
    #     # Hidden Layer 1
    #     model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape,
    #                      name='Discriminator-Hidden-Layer-1'))
    #     model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-1'))
    #
    #     # Hidden Layer 2
    #     model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same',
    #                      name='Discriminator-Hidden-Layer-2'))
    #     model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-2'))
    #
    #     # Hidden Layer 3
    #     model.add(Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
    #                      name='Discriminator-Hidden-Layer-3'))
    #     model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-3'))
    #
    #     # Hidden Layer 4
    #     model.add(Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same',
    #                      name='Discriminator-Hidden-Layer-4'))
    #     model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-4'))
    #
    #     # Flatten and Output Layers
    #     model.add(Flatten(name='Discriminator-Flatten-Layer'))  # Flatten the shape
    #     # Randomly drop some connections for better generalization
    #     model.add(Dropout(0.3, name='Discriminator-Flatten-Layer-Dropout'))
    #     model.add(Dense(1, activation='sigmoid', name='Discriminator-Output-Layer'))  # Output Layer
    #
    #     model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    #                   metrics=['accuracy'])
    #     return model
    #
    # def build_generator(self):
    #     model = Sequential(name="Generator")  # Model
    #
    #     # Hidden Layer 1: Start with 8 x 8 image
    #     n_nodes = 8 * 8 * 256  # number of nodes in the first hidden layer
    #     model.add(Dense(n_nodes, input_dim=self.latent_dim, name='Generator-Hidden-Layer-1'))
    #     model.add(Reshape((8, 8, 256), name='Generator-Hidden-Layer-Reshape-1'))
    #
    #     # Hidden Layer 2: Upsample to 16 x 16
    #     model.add(Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
    #                               name='Generator-Hidden-Layer-2'))
    #     model.add(ReLU(name='Generator-Hidden-Layer-Activation-2'))
    #
    #     # Hidden Layer 3: Upsample to 32 x 32
    #     model.add(Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same',
    #                               name='Generator-Hidden-Layer-3'))
    #     model.add(ReLU(name='Generator-Hidden-Layer-Activation-3'))
    #
    #     # Hidden Layer 4: Upsample to 64 x 64
    #     model.add(Conv2DTranspose(filters=1024, kernel_size=(4, 4), strides=(2, 2), padding='same',
    #                               name='Generator-Hidden-Layer-4'))
    #     model.add(ReLU(name='Generator-Hidden-Layer-Activation-4'))
    #
    #     # Hidden Layer 5: Upsample to 128 x 128
    #     model.add(Conv2DTranspose(filters=1024, kernel_size=(4, 4), strides=(2, 2), padding='same',
    #                               name='Generator-Hidden-Layer-5'))
    #     model.add(ReLU(name='Generator-Hidden-Layer-Activation-5'))
    #
    #     # Hidden Layer 6: Upsample to 256 x 256
    #     model.add(Conv2DTranspose(filters=1024, kernel_size=(4, 4), strides=(2, 2), padding='same',
    #                               name='Generator-Hidden-Layer-6'))
    #     model.add(ReLU(name='Generator-Hidden-Layer-Activation-6'))
    #
    #     # Output Layer (Note, we use 3 filters because we have 3 channels for a color image. Grayscale would have
    #     # only 1 channel)
    #     model.add(
    #         Conv2D(filters=3, kernel_size=(5, 5), activation='tanh', padding='same', name='Generator-Output-Layer'))
    #     return model


    def build_gan(self):
        self.discriminator.trainable = False
        noise_input = layers.Input(shape=(self.latent_dim,))
        gen_img = self.generator(noise_input)
        validity = self.discriminator(gen_img)
        model = Model(noise_input, validity)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        return model

    def train(self, x_train, epochs=5000, batch_size=128):
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Train the discriminator
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_images = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_images = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_images, real)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real)

            # Print the progress
            if epoch % 100 == 0:
                # print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}/{d_loss[1]:.4f}] [G loss: {g_loss:.4f}]")
                print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] "
                      f"[G loss: {g_loss:.4f}]")

    def generate_images(self, num_images=1):
        noise = np.random.normal(-1, 1, (num_images, self.latent_dim))
        return self.generator.predict(noise)
