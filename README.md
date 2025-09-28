# COMP_5313_Assignment_1

## Introduction

### Multi-Layer Preceptron

### Convolutional Neural Network

Unlike the Multi-Layer Preceptron network, a convolutional neural network (CNN) employes steps before a dense layer NN to improve the network's ability to identify and associate features with output classes. A CNN is commonly used when the input object is a 2 or 3 dimensional object divided into pixels, where any pixel is meaningfully related to it's neighbouring pixels. There are 2 main operations that are utilized in CNNs, those being convolutions and pooling. 

#### Convolutions

Convolutions are used to extract features from an input grid that can be used by the model during training, teaching the model to associate certain image features or patterns with different output classes. 

#### Max-Pooling

#### Dropout

While not unique to CNNs, dropout layers are particularly important when dealing with CNNs. A dropout layer radnomly deactivates some number of neurons during an iteration while training. This is commonly done to reduce the risk of overfitting by breaking paths or connections of neurons in the dense layers of a CNN, forcing the network to be more robust. In a CNN, each dense layer will typically have a dropout value between 0.25 and 0.5, meaning that 25% to 50% of the neurons in that particular layer will be randomly deactivated. 
