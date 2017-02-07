# Deep Learning Networks on VERITAS and CTA Image Data

##Description

This repository contains a small library of popular network models configured for use on CTA event images. It also contains several useful scripts for training and analysis.


##General overview of training/testing pipline

The general procedure for training is as follows:

* Generate and store images in directories using imageExtractor and bash scripts
* Run script from models directory with desired arguments to generate a model (.h5) for training
* Use train.py script to train model, saving checkpoints and logs in separate directory
* Visualize results using plot.py and evaluate trained model using evaluate.py


##Files

###Models

#### testNet

##### **Description:** 

Simple CNN for binary classification of telescope data

Credit to: Bryan Kim, example from Francois Chollet (https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)

NOTE: Trained on Theano, image dimension ordering = 'th'

##### Network Architecture Overview

Input: (32,1,120,120)   => batch of 32 depth-1 (grayscale), 120x120 images

Convolution2D(32,3,3)   => 32 filters of size 3x3  
Activation('relu')      => activation using ReLU  
MaxPooling2D(2, 2)      => max-pooling with pools of size 2x2   

Convolution2D(32,3,3)   => 32 filters of size 3x3  
Activation('relu')      => activation using ReLU  
MaxPooling2D(2, 2)      => max-pooling with pools of size 2x2   

Convolution2D(64,3,3)   => 64 filters of size 3x3   
Activation('relu')      => activation using ReLU   
MaxPooling2D(2, 2)      => max-pooling with pools of size 2x2     

Flatten()               => flattens input to linear output vector   
Dense(64)               => standard NN layer with 64 nodes  
Activation('relu')      => activation using ReLU  
Dropout(0.5)            => 50% dropout during train time  
Dense(1)                => standard NN layer with 1 node  
Activation('sigmoid')   => activation using sigmoid function to generate binary classification log probabilities  

##### Properties:

classification: binary

input dim: (1,120,120)

optimizer: nadam

loss function: binary crossentropy

metrics: binary accuracy

weight initialization: glorot normal



#### AlexNet

##### **Description:** 

Based on Alexnet design described in paper. Additional modifications made to network and to preprocessing to suit our dataset. Input dimensions changed to (1,120,120). 

Further details at: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Credit to: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (U. of Toronto)

Credit to: Heuritech, Leonard Blier, https://github.com/heuritech/convnets-keras, Jason Bedford, https://gist.github.com/JBed/c2fb3ce8ed299f197eff

NOTE: Trained on Theano, image dimension ordering = 'th'

##### Network Architecture Overview

Input: (32,1,120,12)

#CONV1
model.add(Convolution2D(96, 11, 11, border_mode='same', input_shape=(1,120,120)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

#CONV2
model.add(Convolution2D(256, 5, 5, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

#CONV3
model.add(Convolution2D(384, 3, 3, border_mode='same'))
model.add(Activation('relu'))

#CONV4
model.add(Convolution2D(384, 3, 3, border_mode='same'))
model.add(Activation('relu'))

#CONV5
model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())

#FC1
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))

#FC2
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))

#FC3
model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

##### Properties:

classification: binary

input dim: (1,120,120)

optimizer: nadam

loss function: binary crossentropy

metrics: binary accuracy

weight initialization: glorot normal


#### VGG16

Credit to: K. Simonyan, A. Zisserman (Very Deep Convolutional Networks for Large-Scale Image Recognition)

Credit to: Gradient Zoo (https://www.gradientzoo.com/commons/keras-vgg-16) w/ some modifications

#### VGG16 (Keras, pretrained)

Credit to: K. Simonyan, A. Zisserman (Very Deep Convolutional Networks for Large-Scale Image Recognition)

Credit to: Francois Chollet (https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py)

#### VGG19

Credit to: K. Simonyan, A. Zisserman (Very Deep Convolutional Networks for Large-Scale Image Recognition)

Credit to: Lorenzo Baraldi (https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d)

#### VGG19 (Keras, pretrained)

Credit to: K. Simonyan, A. Zisserman (Very Deep Convolutional Networks for Large-Scale Image Recognition)

Credit to: Francois Chollet (https://github.com/fchollet/deep-learning-models/blob/master/vgg19.py)

#### ResNet50

Credit to: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Deep Residual Learning for Image Recognition)

Credit to: 

#### GoogLeNet (Inception V1)

Credit to: Christian Szegedy, Wei Liu, Yangqing Jia, et. al. (Going Deeper with Convolutions)

Credit to: 

#### Inception v3

Credit to: Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna (Rethinking the Inception Architecture for Computer Vision) 

Credit to:













###Useful Scripts

#### train.py

#### plot.py

#### evaluate.py

#### train_old.py


