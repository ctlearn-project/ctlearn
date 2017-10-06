import argparse
import os
from shutil import copyfile
import sys

import numpy as np
from PIL import Image

from keras.models import Model
from keras.optimizers import SGD, RMSprop, Nadam
from keras.layers import Dense, Flatten, Input, concatenate
#from keras.layers.merge import Concatenate
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# pSCT image dimensions
image_x_dim= 120
image_y_dim= 120

#samples
training_samples = (358044+185019)
validation_samples = (22842+44757)
#training_samples = 20000
#validation_samples = 10000

# Class directory names
gamma_data_dir = "gamma-diffuse"
proton_data_dir = "proton"

# parse command line arguments

parser = argparse.ArgumentParser(description='Train a given network on data from the indicated directories. Produce log file (.csv) for training run and save plots.')
parser.add_argument('weights', help='path to saved model weights')
parser.add_argument('train_data_dir', help='path to training data directory (containing subdir for each type)')
parser.add_argument('val_data_dir', help='path to validation data directory (containing subdir for each type)')
parser.add_argument('run_name', help='name of run (used to name log file, plot images,etc)')
parser.add_argument('save_dir', help='directory to save run files in (directory must exist)')
parser.add_argument('epochs',help='number of epochs to train for', type=int)
parser.add_argument('--batch_size',help='image generator batch size', default=12, type=int)
#parser.add_argument('--l',help='log results to file',action="store_true")
#parser.add_argument('--c',help='save checkpoints',action="store_true")

args = parser.parse_args()

train_data_path = os.path.abspath(args.train_data_dir)
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


#function to load image into numpy array given filename
#loads png image data into 3-channel format (shape = (1,3,120,120))
#makes 2nd and 3rd channel values all zero
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    img = img.resize((240,240),Image.ANTIALIAS)
    data = np.asarray( img, dtype="uint32" )
    h,w = data.shape
    img_rgb = np.zeros((1,3,h,w), dtype=np.uint32)
    img_rgb[0,0, :, :] =  data
    return img_rgb

def load_empty():
    h = 240
    w = 240
    img_rgb = np.zeros((1,3,h,w), dtype=np.uint32)
    return img_rgb

# Define image generator to yield preprocessed batches of gamma/hadron data
def event_image_generator(path, batch_size):

    #fill dict
    #dict structure:
    #key = event_id
    #value = [dict (key = tel_num, value = full image file name), label]

    dict_events = {}

    for ptype in ['gamma','proton']:
        if ptype == 'gamma':
            data_dir = os.path.join(path, gamma_data_dir)
            label = 1
        else:
            data_dir = os.path.join(path, proton_data_dir)
            label = 0
        for filename in os.listdir(data_dir):
            print("file exists. ")
            if filename.endswith(".png" or ".pn"):
                #get event ID and energy
                evn_id,run_num, energy, impact, tel_num = filename.rsplit(".", 1)[0].split("_")
                event_id=evn_id +'_'+run_num
                event_key = event_id + '_' + ptype
                print("event ID="+ event_id+ "  tel num="+tel_num+"\t")
                print("file is here")
                if event_key in dict_events:
                    dict_events[event_key][0][tel_num] = filename
                else:
                    dict_events[event_key] = [{tel_num:filename},label]
    
    #unique event ids
    total_event_id_count = len(dict_events)
    print("Total event id count: {}".format(total_event_id_count))

    while True:

        event_read_count = 0

        #shuffle list of event ids
        keys = list(dict_events.keys())
        np.random.shuffle(keys)
       
        # Numpy arrays holding flattened training data (image pixel values) (X)
        #each image is a consecutive (1 x 3 x 240 x 240) values
        X_train_T5 = np.empty([batch_size,3,240,240],dtype=np.uint32)
        X_train_T6 = np.empty([batch_size,3,240,240],dtype=np.uint32) 
        X_train_T8 =  np.empty([batch_size,3,240,240],dtype=np.uint32) 
        X_train_T9 = np.empty([batch_size,3,240,240],dtype=np.uint32) 
        X_train_T10 = np.empty([batch_size,3,240,240],dtype=np.uint32) 
        X_train_T11 = np.empty([batch_size,3,240,240],dtype=np.uint32) 
        X_train_T16 = np.empty([batch_size,3,240,240],dtype=np.uint32) 
        X_train_T17 = np.empty([batch_size,3,240,240],dtype=np.uint32) 

        #python list holding training labels (y)
        #1 for gamma
        #0 for proton
        y_train = np.empty([batch_size,],dtype=np.uint32)

        for event_key in keys:
            tels_dict = dict_events[event_key][0]
            event_label = dict_events[event_key][1]
            for tel_num in ['T5','T6','T8','T9','T10','T11','T16','T17']:
                if tel_num in tels_dict:
                    if event_label == 1:
                        img = load_image(os.path.join(path, gamma_data_dir, 
                            tels_dict[tel_num]))
                    else:
                        img = load_image(os.path.join(path, proton_data_dir, 
                            tels_dict[tel_num]))
                else:
                    img = load_empty()

                if tel_num == 'T5':
                    X_train_T5[event_read_count % batch_size,:,:,:] = img
                elif tel_num == 'T6':
                    X_train_T6[event_read_count % batch_size,:,:,:] = img
                elif tel_num == 'T8':
                    X_train_T8[event_read_count % batch_size,:,:,:] = img 
                elif tel_num == 'T9':
                    X_train_T9[event_read_count % batch_size,:,:,:] = img
                elif tel_num == 'T10':
                    X_train_T10[event_read_count % batch_size,:,:,:] = img
                elif tel_num == 'T11':
                    X_train_T11[event_read_count % batch_size,:,:,:] = img     
                elif tel_num == 'T16':
                    X_train_T16[event_read_count % batch_size,:,:,:] = img  
                elif tel_num == 'T17':
                    X_train_T17[event_read_count % batch_size,:,:,:] = img
            y_train[event_read_count % batch_size] = event_label
            event_read_count += 1
            
            if not event_read_count % batch_size:
                input_data = [X_train_T5, X_train_T6, X_train_T8, X_train_T9,
                        X_train_T10, X_train_T11, X_train_T16, X_train_T17]
                labels = y_train
                yield (input_data, labels)

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
inceptionv3 = InceptionV3(
        include_top=False, 
        weights=None, 
        input_shape=(3, image_x_dim*2, image_y_dim*2), 
        pooling=None)
# Add additional layers for connecting to array-level network
inceptionv3_top = Flatten()(inceptionv3.output)
inceptionv3_top = Dense(64, activation='relu', 
        name='telescope_output')(inceptionv3_top)
shared_inception_model = Model(inceptionv3.input, inceptionv3_top)
# Load the pretrained weights, automatically skipping the missing output layer
model_weights_path = os.path.abspath(args.weights)
shared_inception_model.load_weights(model_weights_path, by_name=True)

for layer in shared_inception_model.layers:
        layer.trainable = False

# Define inputs for each telescope
image_1 = Input(shape=(3, 240, 240))
image_2 = Input(shape=(3, 240, 240))
image_3 = Input(shape=(3, 240, 240))
image_4 = Input(shape=(3, 240, 240))
image_5 = Input(shape=(3, 240, 240))
image_6 = Input(shape=(3, 240, 240))
image_7 = Input(shape=(3, 240, 240))
image_8 = Input(shape=(3, 240, 240))

# Create models for each of the telescopes, sharing the same weights for all
print("Defining the model...")
telescope_1 = shared_inception_model(image_1)
telescope_2 = shared_inception_model(image_2)
telescope_3 = shared_inception_model(image_3)
telescope_4 = shared_inception_model(image_4)
telescope_5 = shared_inception_model(image_5)
telescope_6 = shared_inception_model(image_6)
telescope_7 = shared_inception_model(image_7)
telescope_8 = shared_inception_model(image_8)

# Concatenate the telescope outputs into a single array output
array_model = concatenate([telescope_1, telescope_2, telescope_3, telescope_4,
    telescope_5, telescope_6, telescope_7, telescope_8])
array_model = Dense(512, activation='relu', name='dense_1')(array_model)
array_model = Dense(512, activation='relu', name='dense_2')(array_model)
prediction = Dense(1, activation='sigmoid', name='prediction')(array_model)

# Define the model
model = Model([image_1, image_2, image_3, image_4, image_5, image_6, image_7,
    image_8], prediction)

print("Compiling the model...")
model.compile(
        optimizer=Nadam(lr=0.000001),
        loss='binary_crossentropy',
        metrics=['binary_accuracy'])
logger = CSVLogger(os.path.join(run_dir, args.run_name + '.log'),
        separator=',', append=True)
print("Starting the run...")
history = model.fit_generator(
        generator=event_image_generator(args.train_data_dir, args.batch_size),
        steps_per_epoch=int(training_samples/args.batch_size),
        epochs=args.epochs,
        callbacks=[logger,checkpoint],
        validation_data=event_image_generator(args.val_data_dir, 
            args.batch_size),
        validation_steps=int(validation_samples/args.batch_size),
        class_weight='auto')
