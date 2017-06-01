import argparse
import os
from shutil import copyfile
import sys

import numpy as np

from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Nadam
from keras.layers.core import Dense, Flatten
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
#from keras.applications.xception import Xception #Available for TensorFlow only
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# pSCT image dimensions
image_x_dim= 120
image_y_dim= 120
scale_dim=2
#hello
#training_samples = 400
#validation_samples = 50

# parse command line arguments

parser = argparse.ArgumentParser(description='Train a given network on data from the indicated directories. Produce log file (.csv) for training run and save plots.')
parser.add_argument('train_data_dir', help='path to training data directory (containing subdir for each type)')
parser.add_argument('val_data_dir', help='path to validation data directory (containing subdir for each type)')
parser.add_argument('run_name', help='name of run (used to name log file, plot images,etc)')
parser.add_argument('save_dir', help='directory to save run files in (directory must exist)')
parser.add_argument('--model', help='Model. Supported: ResNet50, InceptionV3')
parser.add_argument('--epochs',help='number of epochs to train for', type=int)
parser.add_argument('--optimizer',help='optimizers supported: Adam, Nadam, Adadelta, RMSProp, SGD')
parser.add_argument('--batch_size',help='image generator batch size', default=50, type=int)
parser.add_argument('--lrs',help='scan learning rates (random uniform distribution): log(min) log(max) #lrs', nargs='+', type=float)
parser.add_argument('--lr',help='single learning rate', type=float)

#parser.add_argument('--weights', help='path to saved model weights')
#parser.add_argument('--l',help='log results to file',action="store_true")
#parser.add_argument('--c',help='save checkpoints',action="store_true")

args = parser.parse_args()

mymodel = args.model
suppmod = [ 'ResNet50', 'InceptionV3']
#suppmod = [ 'ResNet50', 'InceptionV3', 'Xception']
if not any(s.lower() == mymodel.lower() for s in suppmod):
    print('Model not supported')
    exit()

myoptimizer = args.optimizer
suppopt = [ 'Adam', 'Nadam', 'Adadelta', 'RMSProp', 'SGD']
if not any(s.lower() == myoptimizer.lower() for s in suppopt):
    print('Optimizer not supported')
    exit()

if args.lr and args.lrs:
    print('--lr and --lrs options are incompatible')
    exit()

#Loading learning rates
runlrs=False
if args.lrs:
    lrs=10**np.random.uniform(args.lrs[0],args.lrs[1],args.lrs[2])
    runlrs=True
else:
    if args.lr:
        lr=args.lr
    elif myoptimizer.lower() == 'adam':
        lr=0.001
    elif myoptimizer.lower() == 'nadam':
        lr=0.002
    elif myoptimizer.lower() == 'adadelta':
        lr=1.0
    elif myoptimizer.lower() == 'rmsprop':
        lr=0.001
    elif myoptimizer.lower()  == 'sgd':
        lr=0.01
    lrs=[lr,lr]

## Load saved weights
#if args.weights is not None:
#    model_weights_path = os.path.abspath(args.weights)
#    model.load_weights(model_weights_path)

# Create required directories
############################

train_data_path = os.path.abspath(args.train_data_dir)
val_data_path = os.path.abspath(args.val_data_dir)

abs_path = os.path.normcase(os.path.abspath(args.save_dir))
run_dir = os.path.join(abs_path, args.run_name)
checkpoint_dir = os.path.join(run_dir, 'checkpoints')
model_dir = os.path.join(run_dir, 'model')

if not os.path.exists(abs_path):
    os.makedirs(abs_path)

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

    
#count number of images in directories
##############################################################

training_samples = sum([len(f) for d, s, f in os.walk(train_data_path)])/2
validation_samples = sum([len(f) for d, s, f in os.walk(val_data_path)])/2

#data augmentation/preprocessing
################################

#processing training data
training_preprocess = ImageDataGenerator(
        horizontal_flip=False,
        vertical_flip=False)

#processing validation data
validation_preprocess = ImageDataGenerator()

#generator for training data
training_generator = training_preprocess.flow_from_directory(
        train_data_path,
        target_size=(image_x_dim*scale_dim, image_y_dim*scale_dim),
#        color_mode='grayscale',
        color_mode='rgb',
        batch_size=args.batch_size,
        class_mode='binary')

#generator for validation data
validation_generator = validation_preprocess.flow_from_directory(
        val_data_path,
        target_size=(image_x_dim*scale_dim, image_y_dim*scale_dim),
#        color_mode='grayscale',
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
accs = []
for lr in lrs:
# Create the model
    if mymodel.lower() == 'resnet50':
        initial_model = ResNet50(
            include_top=False, 
            weights=None, 
            input_shape=(3,image_x_dim*scale_dim, image_y_dim*scale_dim), 
            pooling=None)
    elif mymodel.lower() == 'inceptionv3':
        initial_model = InceptionV3(
            include_top=False, 
            weights=None, 
            input_shape=(3,image_x_dim*scale_dim, image_y_dim*scale_dim), 
            pooling=None)
        
    last = initial_model.output
    x = Flatten()(last)
    prediction = Dense(1, activation='sigmoid')(x)
    model = Model(initial_model.input, prediction)

    if myoptimizer.lower() == 'adam':
        selopt=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    elif myoptimizer.lower() == 'nadam':
        selopt=Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    elif myoptimizer.lower() == 'adadelta':
        selopt=Adadelta(lr=lr, rho=0.95, epsilon=1e-08, decay=0.0)
    elif myoptimizer.lower()  == 'rmsprop':
        selopt=RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0) #lr can be tunned, according to keras.io
    elif myoptimizer.lower()  == 'sgd':
        selopt=SGD(lr=lr)

    model.compile(
        optimizer=selopt,
        loss='binary_crossentropy',
        metrics=['binary_accuracy'])
    
    logger = CSVLogger(os.path.join(run_dir, args.run_name + '-' + mymodel.lower() + '-' + myoptimizer.lower() + '_' + str(lr) + '.log'),
                       separator=',', append=True)
    
    history = model.fit_generator(
        generator=training_generator,
        steps_per_epoch=int(training_samples/args.batch_size),
        epochs=args.epochs,
        callbacks=[logger],
        validation_data=validation_generator,
        validation_steps=int(validation_samples/args.batch_size),
        class_weight='auto')

    accs.append(history.history["val_binary_accuracy"][-1])
    model.save(os.path.join(model_dir, args.run_name + '-' + mymodel.lower() + '-' + myoptimizer.lower() + '_' + str(lr) + '.h5'))
    if runlrs is False:
        break
sorted_lrs = [(lr, acc) for (acc, lr) in sorted(zip(accs, lrs), reverse=True)]
for sorted_lr in sorted_lrs:
    print(sorted_lr)
