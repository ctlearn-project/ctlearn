#Evaluate (calculate the loss,accuracy, and other metrics) of the network (with the existing weights) on a given testing dataset. 
#Hardcoded to work with binary classification and image dimensions (1,120,120)
#

import keras.models
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy
import argparse
import sys
import os
from shutil import copyfile

# parse command line arguments

parser = argparse.ArgumentParser(description='Evaluate a given network on data from the indicated directory. Print outputs.')
parser.add_argument('model', help='path to saved model (ex. model.h5)')
parser.add_argument('test_data_dir', help='path to directory containing images for evaluation')
#parser.add_argument('output_name', help='name used for output log file')
#parser.add_argument('output_dir', help='directory to save output log file in')

args = parser.parse_args()

model_path = os.path.abspath(args.model)
evaluate_data_path = os.path.abspath(args.test_data_dir)

#Image Generator

image_x_dim = 120
image_y_dim = 120

#processing of testing data
test_preprocess = ImageDataGenerator()

#generator for testing data
test_generator = test_preprocess.flow_from_directory(
        test_data_path,
        target_size=(image_x_dim, image_y_dim),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        )

#load saved model
model = load_model(model_path)

# evaluate the model
scores = model.evaluate_generator(test_generator)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
