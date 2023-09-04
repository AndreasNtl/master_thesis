import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scripts.brain_mri_dataset import BrainMriDataset

class BrainCancerClassifier:
    def __init__(self, train_dataloader, val_dataloader, target_size=(64, 64), batch_size=32, epochs=10):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.target_size = target_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.target_size[0], self.target_size[1], 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        train_steps = len(self.train_dataloader)
        val_steps = len(self.val_dataloader)
        history = self.model.fit(self.train_dataloader, epochs=self.epochs, validation_data=self.val_dataloader,
                                 steps_per_epoch=train_steps, validation_steps=val_steps)
        self.plot_training_history(history)
        self.model.save('models/brain_cancer_classifier.keras')
        print("Model saved.")

    def plot_training_history(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()
