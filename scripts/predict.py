#Predict (calculate category classification scores) for a  given testing dataset using a given model with trained weights.
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

parser = argparse.ArgumentParser(description='Predict classification of data from the indicated directory using the given network. Print outputs.')
parser.add_argument('model', help='path to saved model (ex. model.h5)')
parser.add_argument('predict_data_dir', help='path to directory containing images for prediction')
#parser.add_argument('output_name', help='name used for output log file')
#parser.add_argument('output_dir', help='directory to save output log file in')

args = parser.parse_args()

model_path = os.path.abspath(args.model)
predict_data_path = os.path.abspath(args.predict_data_dir)

#Image Generator

image_x_dim = 120
image_y_dim = 120

#processing of prediciton data
predict_preprocess = ImageDataGenerator()

#generator for prediction data
predict_generator = predict_preprocess.flow_from_directory(
        predict_data_path,
        target_size=(image_x_dim, image_y_dim),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        )

#load saved model
model = load_model(model_path)

# predict the dataset using the model
predictions = model.predict_generator(predict_generator)
print(predictions)
