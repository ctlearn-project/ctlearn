import argparse
import os
from shutil import copyfile
import sys

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Dense, Flatten
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#pSCT image dimensions
image_x_dim= 120
image_y_dim= 120

# parse command line arguments

parser = argparse.ArgumentParser(description='Predict on a batch of images and generate plots for the classifier value.')
parser.add_argument('weights', help='path to saved model weights')
#parser.add_argument('model',help='path to saved model')
parser.add_argument('test_data_dir', help='path to testing data directory (containing subdir for each type)')
parser.add_argument('gamma_dir', help='path to gamma test data directory (must contain directory with data)')
parser.add_argument('proton_dir', help='path to proton test data directory (must contain directory with data)')
parser.add_argument('save_dir', help='directory to save plots in')
parser.add_argument('--batch_size',help='image generator batch size', default=20, type=int)

args = parser.parse_args()

test_data_path = os.path.abspath(args.test_data_dir)
gamma_data_path = os.path.abspath(args.gamma_dir)
proton_data_path = os.path.abspath(args.proton_dir)

# Create required directories
############################

save_dir = os.path.normcase(os.path.abspath(args.save_dir))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#count number of images in directories, count number of classes
##############################################################

#data augmentation/preprocessing
################################

#processing training data
test_preprocess = ImageDataGenerator(
        horizontal_flip=False,
        vertical_flip=False)

#generator for training data
#test_generator = test_preprocess.flow_from_directory(
        #test_data_path,
        #target_size=(image_x_dim*2, image_y_dim*2),
        #color_mode='rgb',
        #batch_size=args.batch_size,
        #class_mode='binary')

#generator for gamma data
gamma_test_generator = test_preprocess.flow_from_directory(
        gamma_data_path,
        target_size=(image_x_dim*2, image_y_dim*2),
        color_mode='rgb',
        batch_size=args.batch_size,
        class_mode='binary')

#generator for proton data
proton_test_generator = test_preprocess.flow_from_directory(
        proton_data_path,
        target_size=(image_x_dim*2, image_y_dim*2),
        color_mode='rgb',
        batch_size=args.batch_size,
        class_mode='binary')

# Create the InceptionV3 model
initial_model = InceptionV3(
        include_top=False, 
        weights=None, 
        input_shape=(3, image_x_dim*2, image_y_dim*2), 
        pooling=None)
last = initial_model.output
x = Flatten()(last)
prediction = Dense(1, activation='sigmoid')(x)
model = Model(initial_model.input, prediction)

model.compile(
        optimizer=SGD(lr=0.01),
        loss='binary_crossentropy',
        metrics=['binary_accuracy'])

## Load saved weights
if args.weights is not None:
    model_weights_path = os.path.abspath(args.weights)
    model.load_weights(model_weights_path)

#predict
#result = model.predict_generator(test_generator, 100, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
#filepath = os.path.join(save_dir, 'predict.txt')
#np.savetxt(filepath,result)

result_gamma = model.predict_generator(gamma_test_generator, 50, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
filepath_gamma = os.path.join(save_dir, 'predict_gamma.txt')
result_gamma = 1-result_gamma
np.savetxt(filepath_gamma,result_gamma)

result_proton = model.predict_generator(proton_test_generator, 50, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
filepath_proton = os.path.join(save_dir, 'predict_proton.txt')
result_proton = 1-result_proton
np.savetxt(filepath_proton,result_proton)


