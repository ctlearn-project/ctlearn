
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy
import argparse
import sys
import os
from shutil import copyfile

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# parse command line arguments

parser = argparse.ArgumentParser(description='Train a given network on data from the indicated directories. Produce log file (.csv) for training run and save plots.')
parser.add_argument('model', help='path to saved model (ex. model.h5)')
parser.add_argument('train_data_dir', help='path to training data directory (containing subdir for each type)')
parser.add_argument('val_data_dir', help='path to validation data directory (containing subdir for each type)')
parser.add_argument('run_name', help='name of run (used to name log file, plot images,etc)')
parser.add_argument('save_dir', help='directory to save run files in (directory must exist)')
parser.add_argument('epochs',help='number of epochs to train for', type=int)
parser.add_argument('samples',help='number of images per epoch', type=int, default=122976)
parser.add_argument('--color_mode',help='image generator color mode, grayscale or rgb', default='grayscale')
parser.add_argument('--class_mode', help='-image generator class mode - binary or categorical', default='binary')
parser.add_argument('--batch_size',help='image generator batch size', default=16, type=int)
#parser.add_argument('--l',help='log results to file',action="store_true")
#parser.add_argument('--c',help='save checkpoints',action="store_true")

args = parser.parse_args()

model_path = os.path.abspath(args.model)
train_data_path = os.path.abspath(args.train_data_dir)
val_data_path = os.path.abspath(args.val_data_dir)

#load saved model
model = load_model(model_path)

#create required directories
############################

abs_path = os.path.normcase(os.path.abspath(args.save_dir))
run_dir = os.path.join(abs_path,args.run_name)
checkpoint_dir = os.path.join(run_dir,'checkpoints')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

#move model file to run directory
os.rename(model_path,os.path.join(run_dir, args.run_name + '.h5'))

#count number of images in directories, count number of classes
##############################################################



#data augmentation/preprocessing
################################

image_x_dim= 200
image_y_dim= 200

#processing training data
training_preprocess = ImageDataGenerator(
        #horizontal_flip=True,
        #vertical_flip=True,
        )

#processing validation data
validation_preprocess = ImageDataGenerator()

#generator for training data
training_generator = training_preprocess.flow_from_directory(
        train_data_path,
        target_size=(image_x_dim, image_y_dim),
        color_mode=args.color_mode,
        batch_size=args.batch_size,
        class_mode=args.class_mode,
        )

#generator for validation data
validation_generator = validation_preprocess.flow_from_directory(
        val_data_path,
        target_size=(image_x_dim,image_y_dim),
        color_mode=args.color_mode,
        batch_size=args.batch_size,
        class_mode=args.class_mode,
        )



#train model
############

logger = CSVLogger(os.path.join(run_dir, args.run_name + '.log'),separator=',', append=True)
checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir,args.run_name + '-{epoch:04d}-{val_loss:.5f}.h5'), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
earlystoploss = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, verbose=0, mode='auto')
earlystopacc = EarlyStopping(monitor='val_binary_accuracy', min_delta=0.001, patience=5, verbose=0, mode='auto')
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

history = model.fit_generator(training_generator,samples_per_epoch=args.samples,nb_epoch=args.epochs,callbacks =[logger,checkpoint], validation_data=validation_generator,nb_val_samples=800)

