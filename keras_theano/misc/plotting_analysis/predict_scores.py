import argparse
import os
from shutil import copyfile
import sys

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#pSCT image dimensions
image_x_dim= 120
image_y_dim= 120

# parse command line arguments

parser = argparse.ArgumentParser(description='Predict on two batches of images and output files containing classifier values.')
parser.add_argument('--model',help='path to saved model')
#parser.add_argument('test_data_dir', help='path to testing data directory (containing subdir for each type)')
parser.add_argument('--gamma_dir', help='path to gamma test data directory (must contain directory with data)')
parser.add_argument('--proton_dir', help='path to proton test data directory (must contain directory with data)')
parser.add_argument('--save_dir', help='directory to save predictions in')
parser.add_argument('--batch_size',help='image generator batch size', default=100, type=int)

args = parser.parse_args()

#test_data_path = os.path.abspath(args.test_data_dir)
gamma_data_path = os.path.abspath(args.gamma_dir)
proton_data_path = os.path.abspath(args.proton_dir)

if args.model is not None:
    model_path = os.path.abspath(args.model)
else:
    print('Model not found')
    quit()

# Create required directories
############################

save_dir = os.path.normcase(os.path.abspath(args.save_dir))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#count number of images in directories, count number of classes
##############################################################
num_gamma = sum([len(f) for d, s, f in os.walk(gamma_data_path)])
num_proton = sum([len(f) for d, s, f in os.walk(proton_data_path)])

#data augmentation/preprocessing
################################

#processing training data
test_preprocess = ImageDataGenerator(
        horizontal_flip=False,
        vertical_flip=False)

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

# Load model
model = load_model(model_path)

#predict
#result = model.predict_generator(test_generator, 100, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
#filepath = os.path.join(save_dir, 'predict.txt')
#np.savetxt(filepath,result)

result_gamma = model.predict_generator(gamma_test_generator,num_gamma/args.batch_size , max_q_size=10, workers=1, pickle_safe=False, verbose=0)
filepath_gamma = os.path.join(save_dir, 'predict_gamma_0_1.txt')
result_gamma = 1-result_gamma
np.savetxt(filepath_gamma,result_gamma)

result_proton = model.predict_generator(proton_test_generator, num_proton/args.batch_size, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
filepath_proton = os.path.join(save_dir, 'predict_proton_0_1.txt')
result_proton = 1-result_proton
np.savetxt(filepath_proton,result_proton)
