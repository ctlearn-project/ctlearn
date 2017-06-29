import argparse
import os
from shutil import copyfile
import sys

from preprocessing_fixed.image_fixed import ImageDataGenerator

import numpy as np

from keras.utils.io_utils import HDF5Matrix
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Nadam
from keras.layers.core import Dense, Flatten
from keras.applications.inception_v3 import InceptionV3
#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# parse command line arguments

parser = argparse.ArgumentParser(description='Train a given network on data '
        'from the indicated directories. Produce log file (.csv) for training '
        'run and save plots.')
parser.add_argument('run_name', help='name of run (used to name log file, '
        'plot images,etc)')
parser.add_argument('save_dir', help='directory to save run files in '
        '(directory must exist)')
parser.add_argument('batch_size',help='image generator batch size', type=int)
parser.add_argument('epochs',help='number of epochs to train for', type=int)
parser.add_argument('--weights', help='path to saved model weights',
        default=None)
parser.add_argument('--use_HDF5', help='load data from an HDF5 file',
        default=True)
parser.add_argument('--data_path', help='path to HDF5 data file', default=None)
parser.add_argument('--train_data_dir', help='path to training data directory '
        '(containing subdir for each type)', default=None)
parser.add_argument('--val_data_dir', help='path to validation data directory '
        '(containing subdir for each type)', default=None)

args = parser.parse_args()

if args.train_data_dir:
    train_data_path = os.path.abspath(args.train_data_dir)
if args.val_data_dir:
    val_data_path = os.path.abspath(args.val_data_dir)

# Create required directories
############################

abs_path = os.path.normcase(os.path.abspath(args.save_dir))
run_dir = os.path.join(abs_path, args.run_name)
checkpoint_dir = os.path.join(run_dir, 'checkpoints')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

#data augmentation/preprocessing
################################

# Check that required arguments were specified
hdf5_mode = args.use_HDF5
if hdf5_mode and not args.data_path:
    print("Error: must specify data_path when loading data from HDF5")
    sys.exit()
elif not hdf5_mode and not (args.train_data_dir or args.val_data_dir):
    print("Error: must specify train_data_dir and val_data_dir when loading "
            "images from directories")
    sys.exit()

# Define the generators for preprocessing the data
train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
val_datagen = ImageDataGenerator()

# pSCT image dimensions
image_x_dim= 120
image_y_dim= 120

if hdf5_mode:
    # Load the data
    # TODO: determine these names for our files
    dataset_name = '/0'
    x_path = '/tel_data/5'
    y_path = '/gamma_hadron_label'
    data_x = HDF5Matrix(args.data_path, dataset_name + x_path)
    data_y = HDF5Matrix(args.data_path, dataset_name + y_path)
    
    # Split the data
    train_split = 0.8
    val_split = 0.1
    # test_split = 0.1
    train_data_x = data_x[0:int(train_split*len(data_x))]
    val_data_x = data_x[int(train_split*len(data_x)):int((train_split + val_split)*len(data_x))]
    train_data_y = data_y[0:int(train_split*len(data_y))]
    val_data_y = data_y[int(train_split*len(data_y)):int((train_split + val_split)*len(data_y))]
    
    # Get number of images in dataset
    training_samples = len(train_data_x)
    print("Training samples:", training_samples)
    validation_samples = len(val_data_x)
    print("Validation samples:", validation_samples)

    # Define training and validation generators
    training_generator = train_datagen.flow(train_data_x, train_data_y,
            batch_size=args.batch_size)
    validation_generator = val_datagen.flow(val_data_x, val_data_y,
            batch_size=args.batch_size)
else:    
    # Get number of images in dataset
    training_samples = sum([len(files) for r, d, files in 
        os.walk(train_data_path)])
    #training_samples = (581852+399927)
    print("Training samples:", training_samples)
    validation_samples = sum([len(files) for r, d, files in 
        os.walk(val_data_path)])
    #validation_samples = (72732+49991)
    print("Validation samples:", validation_samples)

    # Define training and validation generators
    training_generator = train_datagen.flow_from_directory(
            train_data_path,
            target_size=(image_x_dim*2, image_y_dim*2),
            color_mode='rgb',
            batch_size=args.batch_size,
            class_mode='binary')
    validation_generator = val_datagen.flow_from_directory(
            val_data_path,
            target_size=(image_x_dim*2, image_y_dim*2),
            color_mode='rgb',
            batch_size=args.batch_size,
            class_mode='binary')

#train model
############

checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir,
    args.run_name + '-{epoch:04d}-{val_loss:.5f}.h5'),
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=True,
    mode='auto')
earlystoploss = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, verbose=0, mode='auto')
earlystopacc = EarlyStopping(monitor='val_binary_accuracy', min_delta=0.001, patience=5, verbose=0, mode='auto')
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

# Create the InceptionV3 model
initial_model = InceptionV3(
        include_top=False, 
        weights=None, 
        input_shape=(3, image_x_dim*2, image_y_dim*2), 
        pooling=None)
last = initial_model.output
x = Flatten()(last)
prediction = Dense(1, activation='sigmoid', name='prediction')(x)
model = Model(initial_model.input, prediction)

## Load saved weights if applicable
if args.weights is not None:
    model_weights_path = os.path.abspath(args.weights)
    model.load_weights(model_weights_path)

model.compile(
        optimizer=Nadam(lr=0.00025),
        loss='binary_crossentropy',
        metrics=['binary_accuracy','binary_crossentropy'])
logger = CSVLogger(os.path.join(run_dir, args.run_name + '.log'),
        separator=',', append=True)
history = model.fit_generator(
        generator=training_generator,
        steps_per_epoch=int(training_samples/args.batch_size),
        epochs=args.epochs,
        callbacks=[logger,checkpoint],
        validation_data=validation_generator,
        validation_steps=int(validation_samples/args.batch_size),
        class_weight='auto')

