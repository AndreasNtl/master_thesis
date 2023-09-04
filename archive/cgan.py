import numpy as np
from keras import layers, models


class CGAN:
    def __init__(self, img_size=28, num_classes=2, latent_dim=100):
        self.img_size = img_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.cgan = self.build_cgan()

    def build_generator(self):
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(1,), dtype='int32')
        label_embedding = layers.Embedding(self.num_classes, self.latent_dim)(label)
        label_embedding = layers.Flatten()(label_embedding)
        model_input = layers.multiply([noise, label_embedding])

        x = layers.Dense(128, activation='relu')(model_input)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(self.img_size * self.img_size * 3, activation='tanh')(x)
        x = layers.Reshape((self.img_size, self.img_size, 3))(x)

        model = models.Model([noise, label], x, name="generator")
        return model

    def build_discriminator(self):
        img = layers.Input(shape=(self.img_size, self.img_size, 3))
        label = layers.Input(shape=(1,), dtype='int32')
        label_embedding = layers.Embedding(self.num_classes, np.prod((self.img_size, self.img_size, 3)))(label)
        label_embedding = layers.Flatten()(label_embedding)
        label_embedding = layers.Reshape((self.img_size, self.img_size, 3))(label_embedding)

        model_input = layers.multiply([img, label_embedding])

        x = layers.Flatten()(model_input)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model([img, label], x, name="discriminator")
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_cgan(self):
        self.discriminator.trainable = False
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(1,), dtype='int32')
        img = self.generator([noise, label])
        valid = self.discriminator([img, label])
        cgan = models.Model([noise, label], valid)
        cgan.compile(loss='binary_crossentropy', optimizer='adam')
        return cgan

    def train(self, x_train, y_train, epochs=5000, batch_size=128):
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        print(epochs)
        for epoch in range(epochs):
            # Train the discriminator
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_images = x_train[idx]
            real_labels = y_train[idx].reshape(-1, 1)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
            gen_images = self.generator.predict([noise, gen_labels])

            d_loss_real = self.discriminator.train_on_batch([real_images, real_labels], real)
            d_loss_fake = self.discriminator.train_on_batch([gen_images, gen_labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
            g_loss = self.cgan.train_on_batch([noise, gen_labels], real)

            # Print the progress
            if epoch % 500 == 0:
                print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

    def generate_images(self, num_images, labels):
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        labels = labels.reshape(-1, 1)
        return self.generator.predict([noise, labels])

#%%
