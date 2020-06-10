import importlib
import logging
import numpy as np
import os
import pkg_resources
import sys
import pandas as pd
import time
import yaml

import tensorflow as tf

def setup_DL1DataReader(config, mode):
    # Parse file list or prediction file list
    if mode in ['train', 'load_only']:
        if isinstance(config['Data']['file_list'], str):
            data_files = []
            with open(config['Data']['file_list']) as f:
                for line in f:
                    line = line.strip()
                    if line and line[0] != "#":
                        data_files.append(line)
            config['Data']['file_list'] = data_files
        if not isinstance(config['Data']['file_list'], list):
            raise ValueError("Invalid file list '{}'. "
                             "Must be list or path to file".format(config['Data']['file_list']))
    else:
        file_list = config['Prediction']['prediction_file_lists'][config['Prediction']['prediction_label']]
        if isinstance(file_list, str):
            data_files = []
            with open(file_list) as f:
                for line in f:
                    line = line.strip()
                    if line and line[0] != "#":
                        data_files.append(line)
            config['Data']['file_list'] = data_files
        if not isinstance(config['Data']['file_list'], list):
            raise ValueError("Invalid prediction file list '{}'. "
                             "Must be list or path to file".format(file_list))

    # Parse list of event selection filters
    event_selection = {}
    for s in config['Data'].get('event_selection', {}):
        s = {'module': 'dl1_data_handler.filters', **s}
        filter_fn, filter_params = load_from_module(**s)
        event_selection[filter_fn] = filter_params
    config['Data']['event_selection'] = event_selection

    # Parse list of image selection filters
    image_selection = {}
    for s in config['Data'].get('image_selection', {}):
        s = {'module': 'dl1_data_handler.filters', **s}
        filter_fn, filter_params = load_from_module(**s)
        image_selection[filter_fn] = filter_params
    config['Data']['image_selection'] = image_selection

    # Parse list of Transforms
    transforms = []
    for t in config['Data'].get('transforms', {}):
        t = {'module': 'dl1_data_handler.transforms', **t}
        transform, args = load_from_module(**t)
        transforms.append(transform(**args))
    config['Data']['transforms'] = transforms

    # Convert interpolation image shapes from lists to tuples, if present
    if 'interpolation_image_shape' in config['Data'].get('mapping_settings',{}):
        config['Data']['mapping_settings']['interpolation_image_shape'] = {
            k: tuple(l) for k, l in config['Data']['mapping_settings']['interpolation_image_shape'].items()}


    # Possibly add additional info to load if predicting to write later
    if mode == 'predict':

        if 'Prediction' not in config:
            config['Prediction'] = {}

        if config['Prediction'].get('save_identifiers', False):
            if 'event_info' not in config['Data']:
                config['Data']['event_info'] = []
            config['Data']['event_info'].extend(['event_id', 'obs_id'])
            if config['Data']['mode'] == 'mono':
                if 'array_info' not in config['Data']:
                    config['Data']['array_info'] = []
                config['Data']['array_info'].append('id')

    return config['Data']

def load_from_module(name, module, path=None, args=None):
    if path is not None and path not in sys.path:
        sys.path.append(path)
    mod = importlib.import_module(module)
    fn = getattr(mod, name)
    params = args if args is not None else {}
    return fn, params

# Define format for TensorFlow dataset
def setup_TFdataset_format(config, example_description, labels):

    config['Input']['output_names'] = [d['name'] for d
                                       in example_description]
    # TensorFlow does not support conversion for NumPy unsigned dtypes
    # other than int8. Work around this by doing a manual conversion.
    dtypes = [d['dtype'] for d in example_description]
    for i, dtype in enumerate(dtypes):
        for utype, stype in [(np.uint16, np.int32), (np.uint32, np.int64)]:
            if dtype == utype:
                dtypes[i] = stype
    config['Input']['output_dtypes'] = tuple(tf.as_dtype(d) for d in dtypes)
    config['Input']['output_shapes'] = tuple(tf.TensorShape(d['shape']) for d
                                             in example_description)
    config['Input']['label_names'] = config['Model']['tasks']

    return config['Input']

# Define input function for TF Estimator
def input_fn(reader, indices, output_names, output_dtypes, output_shapes,
             label_names, mode='train', seed=None, batch_size=1,
             shuffle_buffer_size=None, prefetch_buffer_size=1,
             add_labels_to_features=False):

    def generator(indices):
        for idx in indices:
            yield tuple(reader[idx])

    dataset = tf.data.Dataset.from_generator(generator, output_dtypes,
                                             output_shapes=output_shapes,
                                             args=(indices,))
   
    # Only shuffle the data, when train mode is selected.
    if mode == 'train':
        if shuffle_buffer_size is None:
            shuffle_buffer_size = len(indices)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer_size)

    iterator = dataset.make_one_shot_iterator()

    # Return a batch of features and labels
    example = iterator.get_next()

    features, labels = {}, {}
    for tensor, name in zip(example, output_names):
        dic = labels if name in label_names else features
        dic[name] = tensor
    if mode == 'predict':
        labels = {}

    return features, labels

def get_mc_data(reader, indices, example_description):
    mc_data = {}
    mc_data['tel_pointing'] = reader.tel_pointing
    mc_data['energy_unit'] = 'TeV'
    for i, idx in enumerate(indices):
        for val, des in zip(reader[idx], example_description):
            if des['name'] == 'particletype':
                if i == 0:
                    mc_data['mc_particle'] = []
                mc_data['mc_particle'].append(val)
            elif des['name'] == 'energy':
                if i == 0:
                    mc_data['mc_energy'] = []
                if des['unit'] == 'log(TeV)':
                    mc_data['energy_unit'] = 'log(TeV)'
                    val = np.power(10,val)
                mc_data['mc_energy'].append(val)
            elif des['name'] == 'direction':
                if i == 0:
                    mc_data['mc_altitude'], mc_data['mc_azimuth'] = [],[]
                mc_data['mc_altitude'].append(val[0] + reader.tel_pointing[1])
                mc_data['mc_azimuth'].append(val[1] + reader.tel_pointing[0])
            elif des['name'] == 'impact':
                if i == 0:
                    mc_data['mc_impact_x'], mc_data['mc_impact_y'] = [],[]
                mc_data['mc_impact_x'].append(val[0])
                mc_data['mc_impact_y'].append(val[1])
            elif des['name'] == 'showermaximum':
                if i == 0:
                    mc_data['mc_x_max'] = []
                mc_data['mc_x_max'].append(val)
            elif des['name'] == 'event_id':
                if i == 0:
                    mc_data['event_id'] = []
                mc_data['event_id'].append(val)
            elif des['name'] == 'obs_id':
                if i == 0:
                    mc_data['obs_id'] = []
                mc_data['obs_id'].append(val)
            elif des['name'] == 'tel_id':
                if i == 0:
                    mc_data['tel_id'] = []
                mc_data['tel_id'].append(val)
    return mc_data
