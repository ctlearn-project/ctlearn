# Deep Learning Networks on VERITAS and CTA Image Data

## gammaNNv1

### Files



### **Description:** 

A first attempt at a simple CNN for binary classification of telescope data
images (gamma events vs proton events)

Creates and compiles network, then saves to file.

NOTE: Trained on Theano, image dimension ordering = 'th'

### Network Architecture Overview

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

### Hyperparameters:

optimizer: adam
learning rate: 




