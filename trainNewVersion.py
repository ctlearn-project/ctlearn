
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy
import argparse
import sys

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# parse command line arguments

parser = argparse.ArgumentParser(description='Train a given network on data from the indicated directories. Produce log file (.csv) for training run and save plots.')
parser.add_argument('model', help='path to saved model (ex. model.h5)')
parser.add_argument('train_data_dir', help='path to training data directory (containing subdir for each type)')
parser.add_argument('val_data_dir', help='path to validation data directory (containing subdir for each type)')
parser.add_argument('run_name', help='name of run (used to name log file, plot images, ')
parser.add_argument('epochs',help='number of epochs to train for')
parser.add_argument('samples',help='number of images per epoch')
parser.add_argument('--class_mode', help='-image generator class mode - binary or categorical')
parser.add_argument('--batch_size',help='image generator batch size')
parser.add_argument('--l',help='log results to file',action="store_true")
parser.add_argument('--c',help='save checkpoints',action="store_true")

args = parser.parse_args()

#load saved model
model = load_model(args.model)

#data augmentation/preprocessing
################################

#processing training data
training_preprocess = ImageDataGenerator(
        #horizontal_flip=True,
        #vertical_flip=True,
        )

#processing validation data
validation_preprocess = ImageDataGenerator()

#generator for training data
training_generator = training_preprocess.flow_from_directory(
        args.train_data_dir,
        target_size=(120, 120),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        )

#generator for validation data
validation_generator = validation_preprocess.flow_from_directory(
        args.val_data_dir,
        target_size=(120, 120),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        )

#train model
############

logger = CSVLogger(args.run_name + '.log')
checkpoint = ModelCheckpoint(args.run_name + '{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
earlystoploss = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
earlystopacc = EarlyStopping(monitor='val_binary_accuracy', min_delta=0.001, patience=5, verbose=0, mode='auto')
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

history = model.fit_generator(training_generator,samples_per_epoch=122976,nb_epoch=20,callbacks =[logger,reducelr,earlystopacc], validation_data=validation_generator,nb_val_samples=800)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('accuracy[' + args.run_name + '].png', bbox_inches='tight')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('loss[' args.run_name + '].png', bbox_inches='tight')

#save weights
#############

model.save(args.model)
