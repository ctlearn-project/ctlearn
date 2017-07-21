import argparse
import os
from shutil import copyfile
import sys
import h5py
import random

from configobj import ConfigObj
from validate import Validator
import numpy as np
from PIL import Image
#import tf.train.Feature
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

"""Converts a dataset to tfrecords."""
def convert_to(hdf5_file,energy_bin,name,output_dir,mode): 

    f = h5py.File(hdf5_file,'r')

    if MODE == 'gh_class':
        data = f[energy_bin] 
        labels = data['Gamma_hadron_label']
    elif MODE == 'energy_recon':
        data = f
        labels = data['Energy_bin_label']

    tel_ids = list(data['tel_data'].keys())
    run_numbers = data['run_number']
    event_numbers = data['event_number']

    assert len(run_numbers) == len(event_numbers)
    assert len(labels) == len(event_numbers)
    num_events = len(event_numbers)

    if STORAGE_MODE == 'mapped':
        tel_map = data['tel_map']
        tel_columns = tel_map.attrs['Columns'] 
        assert len(tel_ids) == len(tel_columns)

    tel_counters = {}
    for tel_id in tel_ids:
        tel_counters[tel_id] = 0

    #unique event ids
    print("Total events: {}".format(num_events))

    #all telescope ids
    print("Telescope ids: {}".format(tel_ids))
    
    filename = os.path.join(output_dir, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    if args.mode == 'train':
        start_index = 0
        stop_index = int(num_events*TRAIN_SET) - 1
    elif args.mode == 'validation':
        start_index = int(num_events*TRAIN_SET)
        stop_index = int(num_events*TRAIN_SET) + int(num_events*VALIDATION_SET) - 1
    elif args.mode == 'test':
        start_index = int(num_events*TRAIN_SET) + int(num_events*VALIDATION_SET)
        stop_index = num_events - 1
    else:
        print("Invalid mode choice (choose train,validation,or test)")
        quit(-1)

    for i in range(start_index,stop_index):

        run_number = run_numbers[i]
        event_number = event_numbers[i]
        label = labels[i]
        
        features = {
        'label': _int64_feature(int(labels[i])),
        }

        if STORAGE_MODE == 'mapped':

            tel_trigs = tel_map[i,:]

            for j in range(len(tel_trigs)):
                   
                image_raw_array = np.zeros((1,IMG_CHANNELS,IMAGE_WIDTH,IMAGE_LENGTH), dtype=IMG_DTYPE)
                
                if not tel_trigs[j] == -1:
                    image_raw_array[0,0:HDF5_DATA_CHANNELS, :, :] = data['tel_data'][str(tel_columns[j])][tel_trigs[j]]
                   
                image_raw_string = image_raw_array.tostring()

            features[tel_columns[j]] = _bytes_feature(img_raw_string)

        elif STORAGE_MODE == 'all':
                
                for j in data['tel_data'].keys():

                    image_raw_array = data['tel_data'][j][i,:,:,:]
                    image_raw_string = image_raw_array.tostring()
                    features[tel_columns[j]] = _bytes_feature(img_raw_string)

       
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
    
    writer.close()

if __name__ == '__main__':

    TRAIN_SET = 0.8
    VALIDATION_SET = 0.1
    TEST_SET = 0.1        

    OUTPUT_IMG_CHANNELS = 3
    SCT_IMG_WIDTH = 120
    SCT_IMG_LENGTH = 120

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Takes a directory of gamma images and proton images and convert them into 8 numpy arrays corresponding to the combined training data for each telescope. Also generates and saves a python dict recording the mapping of the rows in the data to event IDs.')
    parser.add_argument('h5_file', help='path to h5 file containing data')
    parser.add_argument('energy_bin', help='energy bin to select data from (0 - n)')
    parser.add_argument('config_file',help='configuration file')
    parser.add_argument('--mode',help='train,validation,test',default='train')
    parser.add_argument('--output_dir',default='.')
    parser.add_argument('output_filename')

    args = parser.parse_args()

    #spec/template for configuration file validation
    config_spec = """
    mode = option('gh_class','energy_recon', default='gh_class')
    storage_mode = option('all','mapped', default='all')
    use_pkl_dict = boolean(default=True)
    [image]
    mode = option('PIXELS_3C','PIXELS_1C','PIXELS_TIMING_2C',PIXELS_TIMING_3C',default='PIXELS_3C')
    scale_factor = integer(min=1,default=2)
    dtype = option('uint32', 'int16', 'int32', 'uint16',default='uint16')
    [telescope]
    type_mode = option('SST', 'SCT', 'LST', 'SST+SCT','SST+LST','SCT+LST','ALL', default='SCT')
    [energy_bins]
    units = option('eV','MeV','GeV','TeV',default='TeV')
    scale = option('linear','log10',default='log10')
    min = float(default=-1.0)
    max = float(default=1.0)
    bin_size = float(default=0.5)
    [preselection_cuts]
    MSCW = tuple(default=list(-2.0,2.0))
    MSCL = tuple(default=list(-2.0,5.0))
    EChi2S = tuple(default=list(0.0,None))
    ErecS = tuple(default=list(0.0,None))
    EmissionHeight = tuple(default=list(0.0,50.0))
    MC_Offset = mixed_list(string,string,float,float,default=list('MCxoff','MCyoff',0.0,3.0))
    NImages = tuple(default=list(3,None))
    dES = tuple(default=list(0.0,None))
    [energy_recon]
        gamma_only = boolean(default=True)
        [[bins]]
        units = option('eV','MeV','GeV','TeV',default='TeV')
        scale = option('linear','log10',default='log10')
        min = float(default=-2.0)
        max = float(default=2.0)
        bin_size = float(default=0.05)
    """

    #Configuration file, load + validate
    spc = config_spec.split('\n')
    config = ConfigObj(args.config_file,configspec=spc)
    validator = Validator()
    val_result = config.validate(validator)

    if val_result:
        print("Config file validated.")

    MODE = config['mode']
    IMG_MODE = config['image']['mode']
    STORAGE_MODE = config['storage_mode']
    IMG_SCALE_FACTOR = config['image']['scale_factor']
    IMG_DTYPE = config['image']['dtype']

    if IMG_MODE == 'PIXELS_3C' or IMG_MODE == 'PIXELS_TIMING_3C':
        HDF5_DATA_CHANNELS = 3
    elif IMG_MODE == 'PIXELS_1C':
        HDF5_DATA_CHANNELS = 1
    elif IMG_MODE == 'PIXELS_TIMING_2C':
        HDF5_DATA_CHANNELS = 2

    convert_to(args.h5_file,args.energy_bin,args.output_filename,args.output_dir,args.mode)
