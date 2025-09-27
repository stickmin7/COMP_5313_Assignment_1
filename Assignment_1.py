import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from medmnist import RetinaMNIST

class convolutional_network():

    def __init__(self):

        self.dataset = RetinaMNIST(split="test", download=True, size=64) # command from https://medmnist.com/ documentation

        self.train_images = np.array(tf.image.rgb_to_grayscale(np.load('/home/henry/.medmnist/retinamnist_64.npz')['train_images'])/255)
        self.train_labels = np.load('/home/henry/.medmnist/retinamnist_64.npz')['train_labels']
        self.train_labels = tf.keras.utils.to_categorical(self.train_labels, num_classes=5)
        self.val_images = np.array(tf.image.rgb_to_grayscale(np.load('/home/henry/.medmnist/retinamnist_64.npz')['val_images'])/255)
        self.val_labels = np.load('/home/henry/.medmnist/retinamnist_64.npz')['val_labels']
        self.val_labels = tf.keras.utils.to_categorical(self.val_labels, num_classes=5)
        self.test_images = np.array(tf.image.rgb_to_grayscale(np.load('/home/henry/.medmnist/retinamnist_64.npz')['test_images'])/255)
        self.test_labels = np.load('/home/henry/.medmnist/retinamnist_64.npz')['test_labels']
        self.test_labels = tf.keras.utils.to_categorical(self.test_labels, num_classes=5)

        self.learning_rate = 0.00001
        self.loss_function = 'CategoricalCrossentropy'
        self.epoch = 1500
        self.batch_size = 128

        self.model = tf.keras.Sequential()

    def view_label_distribution(self):

        print("Training data distribution: ")
        print(np.unique(np.load('/home/henry/.medmnist/retinamnist_64.npz')['train_labels'], return_counts=True))
        print("Validation data distribution: ")
        print(np.unique(np.load('/home/henry/.medmnist/retinamnist_64.npz')['val_labels'], return_counts=True))
        print("Testing data distribution: ")
        print(np.unique(np.load('/home/henry/.medmnist/retinamnist_64.npz')['test_labels'], return_counts=True))

    def define_model(self):

        self.model.add(layers.Input(shape=(64, 64, 1)))

        self.model.add(layers.Conv2D(32, (3, 3,), activation='leaky_relu', padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2D(64, (3, 3,), activation='leaky_relu', padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        #self.model.add(layers.Conv2D(128, (3, 3,), activation='leaky_relu', padding='same'))
        #self.model.add(layers.BatchNormalization())
        #self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))    

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(128, 'leaky_relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))

        self.model.add(layers.Dense(64, 'leaky_relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))

        self.model.add(layers.Dense(32, 'leaky_relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))     

        self.model.add(layers.Dense(5, 'softmax'))

    def train_CNN_model(self):

        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss_function, metrics=['acc'])

        self.model.summary()

        print(self.train_images.shape)
        print(self.train_images.shape)

        return self.model.fit(self.train_images, self.train_labels, validation_data = (self.val_images, self.val_labels), epochs=self.epoch, batch_size=self.batch_size, verbose=2)

    def plot_learning_curves(self, training_accuracy, training_loss, validation_accuracy, validation_loss):

        plt.plot(range(0, self.epoch, 1), training_accuracy, label="training_accuracy")
        plt.plot(range(0, self.epoch, 1), validation_accuracy, label="validation_accuracy")

        plt.title("Accuracy learning curves")
        plt.show()

        plt.plot(range(0, self.epoch, 1), training_loss, label="training_loss")
        plt.plot(range(0, self.epoch, 1), validation_loss, label="validation_loss")

        plt.title("Loss Learning curves")
        plt.show()

def main():
    
    ### Need to add normalization ###

    classification_network = convolutional_network()

    # View label distribution for training, validation, testing

    classification_network.view_label_distribution()

    # define the model

    classification_network.define_model()

    trained_CNN = classification_network.train_CNN_model()

    training_loss = trained_CNN.history['loss']
    validation_loss = trained_CNN.history['val_loss']
    training_accuracy = trained_CNN.history['acc']
    validation_accuracy = trained_CNN.history['val_acc']

    classification_network.plot_learning_curves(training_accuracy, training_loss, validation_accuracy, validation_loss)

    print("model_finished")

if __name__ == "__main__":
    main()