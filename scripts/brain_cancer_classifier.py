import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scripts.brain_mri_dataset import BrainMriDataset
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report


class BrainCancerClassifier:
    def __init__(self, epochs=1000):
        self.epochs = epochs
        self.model = self.build_model()
        self.history = {}


    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
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
    
    def train(self, train_images, train_labels, val_images, val_labels):
        checkpoint = ModelCheckpoint('models/brain_cancer_classifier.keras', save_best_only=True)
        
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        val_images = np.array(val_images)
        val_labels = np.array(val_labels)
        
        # Train the model using NumPy arrays
        self.history = self.model.fit(train_images, train_labels, epochs=self.epochs,
                                 validation_data=(val_images, val_labels),
                                 callbacks=[checkpoint])

        print("Model saved.")


    def plot_training_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()

    def evaluate(self, test_images, test_labels):
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        
        # Evaluate the model using the test data
        loss, accuracy = self.model.evaluate(test_images, test_labels)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

        # Make predictions on the test data
        predictions = self.model.predict(test_images)
        predicted_labels = np.round(predictions)

        # Calculate additional performance metrics
        from sklearn.metrics import classification_report, confusion_matrix

        print("Confusion Matrix:")
        print(confusion_matrix(test_labels, predicted_labels))

        print("\nClassification Report:")
        print(classification_report(test_labels, predicted_labels))

        # Plot confusion matrix
        cm = confusion_matrix(test_labels, predicted_labels)
        self.plot_confusion_matrix(cm, classes=['Class 0', 'Class 1'])

    def plot_confusion_matrix(self, cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()