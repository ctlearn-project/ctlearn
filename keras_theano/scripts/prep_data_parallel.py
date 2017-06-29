import argparse
import os
from shutil import copyfile
import sys
import h5py

import numpy as np
from PIL import Image

# parse command line arguments
parser = argparse.ArgumentParser(description='Take a directory of gamma images and proton images and convert them into 8 numpy arrays corresponding to the combined training data for each telescope. Also generates and saves a python dict recording the mapping of the rows in the data to event IDs.')
parser.add_argument('gamma_data_dir', help='path to gamma data directory (containing subdir for each type)')
parser.add_argument('proton_data_dir', help='path to proton data directory (containing subdir for each type)')
parser.add_argument('save_dir', help='directory to save .hdf5 files in (directory must exist)')
#parser.add_argument('image',help='FOR TESTING')
#parser.add_argument('image2',help='FOR TESTING')

args = parser.parse_args()

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

#function to load empty image array (all zeros)
#shape = (1,3,120,120)
def load_empty_list():
    h = 240
    w = 240
    img_rgb = np.zeros((1,3,h,w), dtype=np.uint32)
    img_rgb_list = np.ravel(img_rgb).tolist()
    #print(len(img_rgb_list))
    return img_rgb_list

def load_empty():
    h = 240
    w = 240
    img_rgb = np.zeros((1,3,h,w), dtype=np.uint32)
    return img_rgb

if __name__ == "__main__":

    #number of iteration (split at ~1500 rows in order to avoid memory errors)
    n = 1
    max_rows = 800
    batch_size = 72

    #fill dicts
    #dict structure: 1 for gamma, 1 for hadron (because of directory structure)
    #each has:
    #key = event_id
    #value = dict (key = tel_num, value = full image file name)
    
    dict_gamma = {}

    for filename in os.listdir(args.gamma_data_dir):
        if filename.endswith(".png"):
            #get event ID and energy
            event_id, energy, impact, tel_num = filename.rsplit(".",1)[0].split("_")

            if event_id in dict_gamma:
                new_dict = dict_gamma[event_id]
                new_dict[tel_num] = filename
                dict_gamma[event_id] = new_dict
            else:
                dict_gamma[event_id] = {tel_num:filename}

    dict_proton = {}

    for filename in os.listdir(args.proton_data_dir):
        if filename.endswith(".png"):
            #get event ID and energy
            event_id, energy, impact, tel_num = filename.rsplit(".",1)[0].split("_")

            if event_id in dict_proton:
                new_dict = dict_proton[event_id]
                new_dict[tel_num] = filename
                dict_proton[event_id] = new_dict
            else:
                dict_proton[event_id] = {tel_num:filename}

    #unique gamma event ids
    print("Gamma events: {}".format(len(dict_gamma)))
    print("Proton events: {}".format(len(dict_proton)))
    total_event_id_count = len(dict_gamma) + len(dict_proton)
    print("Total event id count: {}".format(total_event_id_count))

    event_id_count = 0
    eventid_row_dict = {}

    while(event_id_count < total_event_id_count): 
       
        #python lists holding flattened training data (image pixel values) (X)
        #each image is a consecutive (1 x 3 x 240 x 240) values
        X_train_T5_list = []
        X_train_T6_list = []
        X_train_T8_list = []
        X_train_T9_list = []
        X_train_T10_list = []
        X_train_T11_list = []
        X_train_T16_list = []
        X_train_T17_list = []

        #python list holding training labels (y)
        #1 for gamma
        #0 for proton
        y_train_list = []
        
        row = 0

        for eventid in dict_gamma:
            if eventid in eventid_row_dict:
                continue
            print('{}/{}'.format(event_id_count,total_event_id_count))
            if event_id_count >= total_event_id_count:
                break
            if row == max_rows:
                break
            
            tels_dict = dict_gamma[eventid]
            for tel_num in ['T5','T6','T8','T9','T10','T11','T16','T17']:
                if tel_num in tels_dict:
                    img_list = load_image_list(os.path.join(args.gamma_data_dir,tels_dict[tel_num]))            
                else:
                    img_list = load_empty_list()

                if tel_num == 'T5':
                    X_train_T5_list.extend(img_list)
                elif tel_num == 'T6':
                    X_train_T6_list.extend(img_list)            
                elif tel_num == 'T8':
                    X_train_T8_list.extend(img_list)            
                elif tel_num == 'T9':
                    X_train_T9_list.extend(img_list)            
                elif tel_num == 'T10':
                    X_train_T10_list.extend(img_list)              
                elif tel_num == 'T11':
                    X_train_T11_list.extend(img_list)       
                elif tel_num == 'T16':
                    X_train_T16_list.extend(img_list)                     
                elif tel_num == 'T17':
                    X_train_T17_list.extend(img_list)    

            y_train_list.extend([1])

            eventid_row_dict[eventid] = (n,row,1)
            event_id_count +=1
            row += 1
        
        for eventid in dict_proton:

            if eventid in eventid_row_dict:
                continue
                       
            print('{}/{}'.format(event_id_count,total_event_id_count))

            if event_id_count >= total_event_id_count:
                break

            if row == max_rows:
                break

            tels_dict = dict_proton[eventid]
            for tel_num in ['T5','T6','T8','T9','T10','T11','T16','T17']:
                if tel_num in tels_dict:
                    img_list = load_image_list(os.path.join(args.proton_data_dir,tels_dict[tel_num]))
                    proton_count += 1
                else:
                    img_list = load_empty_list()

                if tel_num == 'T5':
                    X_train_T5_list.extend(img_list)
                elif tel_num == 'T6':
                    X_train_T6_list.extend(img_list)            
                elif tel_num == 'T8':
                    X_train_T8_list.extend(img_list)            
                elif tel_num == 'T9':
                    X_train_T9_list.extend(img_list)            
                elif tel_num == 'T10':
                    X_train_T10_list.extend(img_list)              
                elif tel_num == 'T11':
                    X_train_T11_list.extend(img_list)       
                elif tel_num == 'T16':
                    X_train_T16_list.extend(img_list)                     
                elif tel_num == 'T17':
                    X_train_T17_list.extend(img_list)    

            y_train_list.extend([0])

            eventid_row_dict[eventid] = (n,row,0)
            event_id_count +=1
            row += 1

        #print(X_train_T5_list)

        X_train_T5 = np.reshape(X_train_T5_list,newshape=(row,3,240,240))
        np.save(os.path.join(args.save_dir,'X_train_T5_' + str(n) + '.npy'),X_train_T5)
        X_train_T5 = np.empty((0,3,240,240),dtype=np.uint32)

        X_train_T6 = np.reshape(X_train_T6_list,newshape=(row,3,240,240))
        np.save(os.path.join(args.save_dir,'X_train_T6_' + str(n) + '.npy'),X_train_T6)
        X_train_T6 = np.empty((0,3,240,240),dtype=np.uint32)

        X_train_T8 = np.reshape(X_train_T8_list,newshape=(row,3,240,240))
        np.save(os.path.join(args.save_dir,'X_train_T8_' + str(n) + '.npy'),X_train_T8)
        X_train_T8 = np.empty((0,3,240,240),dtype=np.uint32)

        X_train_T9 = np.reshape(X_train_T9_list,newshape=(row,3,240,240))
        np.save(os.path.join(args.save_dir,'X_train_T9_' + str(n) + '.npy'),X_train_T9)
        X_train_T9 = np.empty((0,3,240,240),dtype=np.uint32)

        X_train_T10 = np.reshape(X_train_T10_list,newshape=(row,3,240,240))
        np.save(os.path.join(args.save_dir,'X_train_T10_' + str(n) + '.npy'),X_train_T10)
        X_train_T10 = np.empty((0,3,240,240),dtype=np.uint32)

        X_train_T11 = np.reshape(X_train_T11_list,newshape=(row,3,240,240))
        np.save(os.path.join(args.save_dir,'X_train_T11_' + str(n) + '.npy'),X_train_T11)
        X_train_T11 = np.empty((0,3,240,240),dtype=np.uint32)

        X_train_T16 = np.reshape(X_train_T16_list,newshape=(row,3,240,240))
        np.save(os.path.join(args.save_dir,'X_train_T16_' + str(n) + '.npy'),X_train_T16) 
        X_train_T16 = np.empty((0,3,240,240),dtype=np.uint32)
    
        X_train_T17 = np.reshape(X_train_T17_list,newshape=(row,3,240,240))
        np.save(os.path.join(args.save_dir,'X_train_T17_' + str(n) + '.npy'),X_train_T17)
        X_train_T17 = np.empty((0,3,240,240),dtype=np.uint32)

        y_train = np.array(y_train_list)
        np.save(os.path.join(args.save_dir,'y_train_' + str(n) + '.npy'),y_train)
        y_train = np.empty((0,),dtype=np.uint32)   

        row = 0
        n += 1

    dict_file_name = 'eventid_row_dict.pkl'
    with open(os.path.join(args.save_dir,dict_file_name), 'wb') as f:
        pickle.dump(eventid_row_dict, f, pickle.HIGHEST_PROTOCOL)


