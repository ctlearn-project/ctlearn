import importlib
import logging
import os
import pkg_resources
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml


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
        if file_list.endswith(".txt"):
            data_files = []
            with open(file_list) as f:
                for line in f:
                    line = line.strip()
                    if line and line[0] != "#":
                        data_files.append(line)
            config['Data']['file_list'] = data_files
        elif file_list.endswith(".h5"):
            config['Data']['file_list'] = [file_list]
        if not isinstance(config['Data']['file_list'], list):
            raise ValueError("Invalid prediction file list '{}'. "
                             "Must be list or path to file".format(file_list))

    data_format = config.get('Data_format', 'stage1')
    if data_format == 'dl1dh':
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
    config['Input']['label_names'] = config['Model']['tasks']

    return config['Input']

# Define input function for TF Estimator
def input_fn(reader, indices, output_names, output_dtypes,
             label_names, shuffle_and_repeat=False, num_epochs=None, seed=None,
             batch_size=1, prefetch_to_device=None,
             add_labels_to_features=False):

    dataset = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle_and_repeat:
        dataset = dataset.shuffle(buffer_size=len(indices), seed=seed,
                                      reshuffle_each_iteration=True)
        dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(lambda x: tf.py_function(func=reader.__getitem__,
                                                   inp=[x],
                                                   Tout=output_dtypes),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    if prefetch_to_device is not None:
        dataset = dataset.apply(
            tf.data.experimental.prefetch_to_device(**prefetch_to_device))

    iterator = dataset.make_one_shot_iterator()

    # Return a batch of features and labels
    example = iterator.get_next()

    features, labels = {}, {}
    for tensor, name in zip(example, output_names):
         dic = labels if name in label_names else features
         dic[name] = tensor

    if add_labels_to_features:  # for predict mode
         features['labels'] = labels

    return features, labels
