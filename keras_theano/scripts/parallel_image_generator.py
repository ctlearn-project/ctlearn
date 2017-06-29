import argparse
import os
from shutil import copyfile
import sys
import h5py
import random

import numpy as np
from PIL import Image

# parse command line arguments
parser = argparse.ArgumentParser(description='Takes a directory of gamma images and proton images and convert them into 8 numpy arrays corresponding to the combined training data for each telescope. Also generates and saves a python dict recording the mapping of the rows in the data to event IDs.')
parser.add_argument('gamma_data_dir', help='path to gamma data directory (containing subdir for each type)')
parser.add_argument('proton_data_dir', help='path to proton data directory (containing subdir for each type)')
#parser.add_argument('save_dir', help='directory to save .hdf5 files in (directory must exist)')
#parser.add_argument('image',help='FOR TESTING')
#parser.add_argument('image2',help='FOR TESTING')

args = parser.parse_args()

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

'''
#function to load image into numpy array given filename
#loads png image data into 3-channel format (shape = (1,3,120,120))
#makes 2nd and 3rd channel values all zero
def load_image_list( infilename ) :
    img = Image.open( infilename )
    img.load()
    img = img.resize((240,240),Image.ANTIALIAS)
    data = np.asarray( img, dtype="uint32" )
    h,w = data.shape
    img_rgb = np.zeros((1,3,h,w), dtype=np.uint32)
    img_rgb[0,0, :, :] =  data
    img_rgb_list = np.ravel(img_rgb).tolist()
    #print(len(img_rgb_list))
    return img_rgb_list

#function to load empty image array (all zeros)
#shape = (1,3,120,120)
def load_empty_list():
    h = 240
    w = 240
    img_rgb = np.zeros((1,3,h,w), dtype=np.uint32)
    img_rgb_list = np.ravel(img_rgb).tolist()
    #print(len(img_rgb_list))
    return img_rgb_list

'''

if __name__ == "__main__":

    #number of samples to return in 1 batch (constrained by memory)
    batch_size = 72

    #fill dict
    #dict structure:
    #key = event_id
    #value = [dict (key = tel_num, value = full image file name), label]
    
    dict_events = {}

    for ptype in ['gamma','proton']:
        if ptype == 'gamma':
            data_dir = args.gamma_data_dir
            label = 1
        else:
            data_dir = args.proton_data_dir
            label = 0

        for filename in os.listdir(args.gamma_data_dir):
            if filename.endswith(".png"):
                #get event ID and energy
                event_id, energy, impact, tel_num = filename.rsplit(".",1)[0].split("_")

                if event_id in dict_events:
                    dict_events[event_id][0][tel_num] = filename
                else:
                    dict_events[event_id] = [{tel_num:filename},label]
    
    #unique event ids
    total_event_id_count = len(dict_events)
    print("Total event id count: {}".format(total_event_id_count))

    while True:

        event_read_count = 0

        #shuffle list of event ids
        keys = dict_event.keys()
        random.shuffle(keys)
       
        #python lists holding flattened training data (image pixel values) (X)
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

        for event_id in keys:
            
            event_label = dict_events[event_id][1]
            tels_dict = dict_events[event_id][0]
            for tel_num in ['T5','T6','T8','T9','T10','T11','T16','T17']:
                if tel_num in tels_dict:
                    if event_label == 1:
                        img = load_image(os.path.join(args.gamma_data_dir,tels_dict[tel_num]))
                    else:
                        img = load_image(os.path.join(args.proton_data_dir,tels_dict[tel_num]))
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

            y_train_list.extend([event_label])

            event_read_count +=1
       
            if not event_read_count % batch_size:

                input_data = [X_train_T5,X_train_T6,X_train_T8,X_train_T9,X_train_T10,X_train_T11,X_train_T16,X_train_T17]
                labels = y_train

                yield (input_data,labels)

