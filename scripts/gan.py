import numpy as np
from keras import layers, Sequential
from keras.layers import Dense, Reshape, Conv2DTranspose, ReLU, Conv2D, LeakyReLU, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime


class SimpleGAN:
    def __init__(self, img_size, latent_dim=100):
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.channels = 3
        self.discriminator = self.build_discriminator(in_shape=(img_size, img_size, 3))
        self.generator = self.build_generator()
        self.gan = self.build_gan()
        self.d_losses = []  # To store discriminator losses
        self.g_losses = []  # To store generator losses
        self.generated_images = []
        self.discriminator.summary()  # Print discriminator architecture
        self.generator.summary()      # Print generator architecture

        # Get the current timestamp
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.generator_path = 'models/generator_model.keras'
        self.discriminator_path = 'models/discriminator_model.keras'


        # Load models if they exist
        self.models_loaded = False  # Flag to track if models have been loaded
        if os.path.exists(self.discriminator_path) and os.path.exists(self.generator_path):
            self.load_models()
            self.models_loaded = True 

    def build_discriminator(self, in_shape=(64, 64, 3)):
        model = Sequential(name="Discriminator")  # Model

        # Hidden Layer 1
        model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape,
                         name='Discriminator-Hidden-Layer-1'))
        model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-1'))

        # Hidden Layer 2
        model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                         name='Discriminator-Hidden-Layer-2'))
        model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-2'))

        # Hidden Layer 3
        model.add(Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                         name='Discriminator-Hidden-Layer-3'))
        model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-3')) 

        # Flatten and Output Layers
        model.add(Flatten(name='Discriminator-Flatten-Layer'))  # Flatten the shape
        # Randomly drop some connections for better generalization
        model.add(Dropout(0.3, name='Discriminator-Flatten-Layer-Dropout'))
        model.add(Dense(1, activation='sigmoid', name='Discriminator-Output-Layer'))  # Output Layer 

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


    def build_gan(self):
        self.discriminator.trainable = False
        noise_input = layers.Input(shape=(self.latent_dim,))
        gen_img = self.generator(noise_input)
        validity = self.discriminator(gen_img)
        model = Model(noise_input, validity)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        return model

    def train(self, x_train, epochs=5000, batch_size=128):

        if self.models_loaded:  # Check if models are already loaded
            print("Models are already loaded. Training skipped.")
            return

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Train the discriminator
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_images = x_train[idx]

            noise = np.random.normal(-1, 1, (batch_size, self.latent_dim))
            fake_images = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_images, real)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator
            noise = np.random.normal(-1, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real)

            # Store losses for visualization
            self.d_losses.append(d_loss[0])
            self.g_losses.append(g_loss)

            # Print the progress
            if epoch % 100 == 0:
                print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] "
                      f"[G loss: {g_loss:.4f}]")

            if epoch % 1000 == 0:
                self.save_models()

        self.save_models()


    def generate_images(self, num_images=1, output_dir='generated_images'):
        noise = np.random.normal(-1, 1, (num_images, self.latent_dim))
        self.generated_images = self.generator.predict(noise)
        # Save the generated images
        for i, image in enumerate(self.generated_images):
            image_filename = os.path.join(output_dir, f'generated_image_{i}_{self.timestamp}.png')
            plt.imsave(image_filename, (image + 1) / 2)

        return self.generated_images


    def evaluate_discriminator(self, x_test):
        real_labels = np.ones((x_test.shape[0], 1))
        fake_labels = np.zeros((x_test.shape[0], 1))

        # Convert PyTorch tensor to NumPy array
        x_test_numpy = x_test.numpy()

        # Evaluate discriminator on real images
        real_accuracy = self.discriminator.evaluate(x_test_numpy, real_labels, verbose=0)

        # Evaluate discriminator on generated images
        generated_images = self.generate_images(num_images=x_test.shape[0])
        fake_accuracy = self.discriminator.evaluate(generated_images, fake_labels, verbose=0)

        print(f"Real Images - Loss: {real_accuracy[0]:.4f}, Accuracy: {100 * real_accuracy[1]:.2f}%")
        print(f"Generated Images - Loss: {fake_accuracy[0]:.4f}, Accuracy: {100 * fake_accuracy[1]:.2f}%")

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label="Discriminator Loss")
        plt.plot(self.g_losses, label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("GAN Training Losses")
        plt.legend()
        plt.savefig(f"./visualizations/losses_{self.timestamp}.png")
        plt.show()
        plt.close()

    def plot_real_vs_fake(self, x_test, epoch, num_examples=16, figsize=(8, 4)):
        real_labels = np.ones((num_examples, 1))
        fake_labels = np.zeros((num_examples, 1))

        real_images = x_test[:num_examples]

        plt.figure(figsize=figsize)
        for i in range(num_examples):
            plt.subplot(2, num_examples, i + 1)
            plt.imshow((real_images[i] + 1) / 2)
            plt.title("Real")
            plt.axis("off")

            plt.subplot(2, num_examples, num_examples + i + 1)
            plt.imshow((self.generated_images[i] + 1) / 2)
            plt.title("Fake")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"./visualizations/real_vs_fake_epoch_{epoch}_{num_examples}_{self.timestamp}.png")
        plt.show()
        plt.close()

    def save_models(self):
        self.discriminator.save(self.discriminator_path)
        self.generator.save(self.generator_path)
        np.save('models/d_losses.npy', self.d_losses)
        np.save('models/g_losses.npy', self.g_losses)
        print("Models and loss histories saved.")


    def load_models(self):
        self.discriminator = tf.keras.models.load_model(self.discriminator_path)
        self.generator = tf.keras.models.load_model(self.generator_path)
        self.gan = self.build_gan()  # Rebuild the GAN with the loaded models

        # Load loss histories
        self.d_losses = np.load('models/d_losses.npy').tolist()
        self.g_losses = np.load('models/g_losses.npy').tolist()

        print("Models loaded.")


