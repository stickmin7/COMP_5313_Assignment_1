# COMP_5313_Assignment_1

## Introduction

### RetinaMNIST Dataset

The RetinaMNIST datset is part of the larger MEdMNIST datset, consisitng of 18 standarized 2D and 3D medical image datasets [1, 2]. For this assignment, only the RetinaMNIST datset is used, which consists of 1600 retina fundus images, split into training, validation and testing sets. Unlike most of the other datsets which are classification problems, the RetinaMNIST involes ordinal regression for 5 different severity levels of diabetic retinopathy [1]. Ordinal regression is similar to classification, but the values in this case exist on a scale and the ordering of the classes on the scale is important (unlike regular classification problems, where the classes to be predicted are discrete and unordered). 

The goal of this assignment is to compare the preformance of mutlilayer preceptron (MLP) networks vs convolutional neural networks (CNNs). 

### Multi-Layer Preceptron

#### Network Layers and Neuron Operation

In a multi-layer preceptron network, layers are classified as input, output, or hidden, depending on the role of the neurons in the layer. The neurons in the input and output layers contain the input features and output classes respectively. The neurons in the hidden layers, and the connections to and from these neurons, are what are primarly adjusted by the network during training to learn from the information feed by the input neurons. Generally, each neuron in a layer is connected to all the neurons in the layer previous and the following layer, though this can be changed. 

When training, the inout of each neuron is given as the sum of the inputs times their respective weights before a bias is added. This entire sum is then input into an activation function before the final value is output into the next layer of neurons (or to the program if this is the output layer). After all the data in a single epoch has been input and output by the network, it uses a loss function to compare the predicted values to the actual values. The neural network will adjust the weights and biases in the network to optimize (minimize) the loss function and improve the accuracy of the network's predictions. 

### Convolutional Neural Network

Unlike the Multi-Layer Preceptron network, a convolutional neural network (CNN) employes steps before a dense layer NN to improve the network's ability to identify and associate features with output classes. A CNN is commonly used when the input object is a 2 or 3 dimensional object divided into pixels, where any pixel is meaningfully related to it's neighbouring pixels. There are 2 main operations that are utilized in CNNs, those being convolutions and pooling. 

#### Convolutions

Convolutions are used to extract features from an input grid that can be used by the model during training, teaching the model to associate certain image features or patterns with different output classes. A kernel matrix (usually of size 3x3 or 5x5 pixels) is slid across the input map. As the kernel is slid across the map, dot product is preformed between the input and kernel maps, creating a new output map with a feature or pattern extracted from the input map. This would result in a downsize of the output map compared to the input map, which is not nessesarily desireable. This issue can be resolved using a padding method, where zeros are added around the edges of the input map. This allows the convolution operation to be comducted at the egdes of the input map and not reduce the dimensions of the map.

#### Max-Pooling

Unlike convolutions, this operation is meant to downsize the feature maps created in the convolution step. A kernel is slid across the feature maps, but now a summarizing operation, such as average or max pooling is preformed to produce only a single value for each position. The goal of the pooling operations is to reduce the size of the feature maps while still retaining the patterns or features found by the convolution layers. Two of the more common methods to preform pooling are max pooling and average pooling, where either only the max value for a region of the feature map is retained, or the average of a region of the feature map is retained. 

#### Dropout

While not unique to CNNs, dropout layers are particularly important when dealing with CNNs. A dropout layer radnomly deactivates some number of neurons during an iteration while training. This is commonly done to reduce the risk of overfitting by breaking paths or connections of neurons in the dense layers of a CNN, forcing the network to be more robust. In a CNN, each dense layer will typically have a dropout value between 0.25 and 0.5, meaning that 25% to 50% of the neurons in that particular layer will be randomly deactivated. 

## Code 

### General 

Main function:

      --> convolutional_network/multi_layer_network class   

            --> init function   

                  --> Opens the RetinaMNIST dataset and loads the images and labels for the training, validation and testing datasets. The learning rate, loss function, epoch and batch size are defined in this function as well. 

            --> view_label_distribution function

                  --> This displays the size of each of the datasets.

            --> define_model function

                  --> This function for both classes defines the model archeture using the Keras python libary.

            --> train_CNN_model function

                  --> This function complies the model using the optimizer, learning rate and loss specified by the user. Class weights are also calculated in this function to help mitigate the class imbalance of the retina dataset. The model is finally fit to the data and returns the fitted model.

            --> test_model function

                  --> Most of the analysis of the network is done in this function. The Keras predict function is used to test the model. The class with the output neuron containing the highest value is taken as the model prediction and compared against the test labels.

            --> plot_learning_curves function

                  --> Short function to plot the training and validation loss curves

### Network

### Hyperparameters

## Results

## Discussion

## Conclusion

## References 

[1] Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023.
                            
[2] Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.
