from PIL import ImageOps
from keras import layers
import keras
import random


class ImageDataAugmentation:
    def __init__(self, image):
        self.image = image

    def random_rotate(self, min_angle=-45, max_angle=45):
        angle = random.uniform(min_angle, max_angle)
        self.image = self.image.rotate(angle)
        return self

    def random_horizontal_flip(self, p=0.5):
        if random.random() < p:
            self.image = ImageOps.mirror(self.image)
        return self

    def random_vertical_flip(self, p=0.5):
        if random.random() < p:
            self.image = ImageOps.flip(self.image)
        return self

    def resize_and_rescale(self):
        IMG_SIZE = 128
        keras.Sequential([
            layers.Resizing(IMG_SIZE, IMG_SIZE),
            layers.Rescaling(1./255)
        ])
        return self

    def apply(self):
        return self.image

    def save(self, output_path):
        self.image.save(output_path)
