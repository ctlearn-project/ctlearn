# Deep Learning Networks on VERITAS and CTA Image Data

##Models

### gammaNNv1

#### **Description:** 

Simple CNN for binary classification of telescope data

Credit to: 

NOTE: Trained on Theano, image dimension ordering = 'th'

#### Network Architecture Overview

Input: (32,1,120,120) => batch of 32 depth-1 (grayscale), 120x120 images

Convolution2D(32,3,3) => 32 filters of size 3x3  
Activation('relu') => activation using ReLU  
MaxPooling2D(2, 2) => max-pooling with pools of size 2x2   

Convolution2D(32,3,3) => 32 filters of size 3x3  
Activation('relu') => activation using ReLU  
MaxPooling2D(2, 2) => max-pooling with pools of size 2x2   

Convolution2D(64,3,3) => 64 filters of size 3x3   
Activation('relu') => activation using ReLU   
MaxPooling2D(2, 2) => max-pooling with pools of size 2x2     

Flatten() => flattens input to linear output vector   
Dense(64) => standard NN layer with 64 nodes  
Activation('relu') => activation using ReLU  
Dropout(0.5) => 50% dropout during train time  
Dense(1) => standard NN layer with 1 node  
Activation('sigmoid') => activation using sigmoid function to generate binary classification log probabilities  

#### Hyperparameters:

optimizer: adam  
learning rate:   

### AlexNet

Additional modifications made to network and to preprocessing to suit our dataset.

Further details at: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Credit to: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (U. of Toronto)

Credit to: Heuritech, Leonard Blier, https://github.com/heuritech/convnets-keras

### GoogLeNet (Inception V1)

Credit to:

### VGG16

Credit to:

### VGG19

Credit to:

### ResNet50

Credit to:

### Inception v3

Credit to:















##Useful Scripts

### train.py

### plot.py

### evaluate.py





