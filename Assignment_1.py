import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from medmnist import RetinaMNIST

class convolutional_network():

    def __init__(self):

        self.dataset = RetinaMNIST(split="test", download=True, size=64) # command from https://medmnist.com/ documentation

        self.train_images = np.load('/home/henry/.medmnist/retinamnist_64.npz')['train_images']
        self.train_labels = np.load('/home/henry/.medmnist/retinamnist_64.npz')['train_labels']
        self.train_labels = tf.keras.utils.to_categorical(self.train_labels, num_classes=5)
        self.val_images = np.load('/home/henry/.medmnist/retinamnist_64.npz')['val_images']
        self.val_labels = np.load('/home/henry/.medmnist/retinamnist_64.npz')['val_labels']
        self.val_labels = tf.keras.utils.to_categorical(self.val_labels, num_classes=5)
        self.test_images = np.load('/home/henry/.medmnist/retinamnist_64.npz')['test_images']
        self.test_labels = np.load('/home/henry/.medmnist/retinamnist_64.npz')['test_labels']
        self.test_labels = tf.keras.utils.to_categorical(self.test_labels, num_classes=5)

        self.learning_rate = 0.0001
        self.loss_function = 'categorical_crossentropy'
        self.epoch = 50
        self.batch_size = 16

    def input_layer(self):

        return layers.Input(shape=(64, 64, 3)) 

    def convolution_layers(self, layer, kernels, dim):
        
        return layers.Conv2D(kernels, dim, activation='relu', padding='same')(layer)

    def max_pooling_layers(self, layer, pool_x, pool_y):

        return layers.MaxPooling2D(pool_size=(pool_x, pool_y), padding='same')(layer)
    
    def dropout_layers(self, layer):
        
        return layers.Dropout(0.4)(layer)
    
    def batch_normalization(self, layer):

        return layers.BatchNormalization()(layer)

    def dense_layers(self, layer, neurons, activation_func):
        
        return layers.Dense(neurons, activation=activation_func)(layer)        

    def train_CNN_model(self, input_layer, output_layer):

        model = models.Model(input_layer, output_layer)

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss_function, metrics=['acc'])

        model.summary()

        return model.fit(self.train_images, self.train_labels, validation_data = (self.val_images, self.val_labels), epochs=self.epoch, batch_size=self.batch_size, verbose=2)

def main():
    
    ### Need to add normalization ###

    classification_network = convolutional_network()

    # define the model

    input_layer = classification_network.input_layer()
    
    # convolutions and max poolings

    conv_layer = classification_network.convolution_layers(input_layer, 64, 5)
    normalization = classification_network.batch_normalization(conv_layer)
    conv_layer = classification_network.convolution_layers(normalization, 128, 3)
    normalization = classification_network.batch_normalization(conv_layer)
    max_pooling = classification_network.max_pooling_layers(normalization, 2, 2)

    # Convert 2D matrices into 1D feature matrix

    flattened_layer = layers.Flatten()(max_pooling)

    # Dense layers

    dense_layer = classification_network.dense_layers(flattened_layer, 512, 'sigmoid')
    dropout = classification_network.dropout_layers(dense_layer)
    dense_layer = classification_network.dense_layers(dropout, 256, 'sigmoid')
    dropout = classification_network.dropout_layers(dense_layer)
    dense_layer = classification_network.dense_layers(dropout, 128, 'sigmoid')
    dropout = classification_network.dropout_layers(dense_layer)
    output_layer = classification_network.dense_layers(dropout, 5, 'sigmoid')

    # Encoded data to fix input shape error

    trained_CNN = classification_network.train_CNN_model(input_layer, output_layer)

    print(trained_CNN.history)

    print("model_finished")

if __name__ == "__main__":
    main()