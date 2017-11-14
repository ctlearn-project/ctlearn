import configparser
import os
import shutil
import sys
import time

# Disable info and warning messages (not error messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import tables

# Add parent directory to pythonpath to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from models.variable_input_model import variable_input_model

def load_HDF5_data(filename, index, metadata, mode='TRAIN'):

    # Read the data at the given table and index from the file
    f = tables.open_file(filename, mode='r')
    if mode == 'TRAIN':
        table = f.root.E0.Events_Training
    elif mode == 'VALID':
        table = f.root.E0.Events_Validation
    else:
        raise ValueError("Mode must be 'TRAIN' or 'VALID'")
    record = table.read(index, index + 1)
    
    telescope_ids = metadata['telescope_ids']
    image_indices = record['tel_map'][0]
    telescope_images = []
    for telescope_id, image_index in zip(telescope_ids, image_indices):
        if image_index == -1:
            # Telescope did not trigger. Its outputs will be dropped
            # out, so input is arbitrary. Use an empty array for
            # efficiency.
            telescope_images.append(np.empty(metadata['image_shape']))
        else:
            telescope_table = f.root.E0._f_get_child(telescope_id)
            telescope_images.append(telescope_table[image_index])
    telescope_images = np.stack(telescope_images).astype(np.float32)
    
    # Get binary values indicating whether each telescope triggered
    telescope_triggers = np.array([0 if i < 0 else 1 for i in image_indices],
            dtype=np.int8)
    
    # Get classification label by converting CORSIKA particle code
    gamma_hadron_label = record['gamma_hadron_label'][0]
    if gamma_hadron_label == 0: # gamma ray
        gamma_hadron_label = 1
    elif gamma_hadron_label == 101: # proton
        gamma_hadron_label = 0
    
    f.close()
    
    return [telescope_images, telescope_triggers, gamma_hadron_label]

def load_HDF5_auxiliary_data(filename):
    
    f = tables.open_file(filename, mode='r')
    telescope_positions = []
    for row in f.root.Tel_Table.iterrows():
        telescope_positions.append(row["tel_x"])
        telescope_positions.append(row["tel_y"])
    f.close()
    auxiliary_data = {
        'telescope_positions': np.array(telescope_positions, dtype=np.float32)
        }
    return auxiliary_data

def load_HDF5_metadata(filename):
   
    f = tables.open_file(filename, mode='r')
    num_training_events = f.root.E0.Events_Training.shape[0]
    num_validation_events = f.root.E0.Events_Validation.shape[0]
    # List of telescope IDs ordered by mapping index
    telescope_ids = ["T" + str(row["tel_id"]) for row 
            in f.root.Tel_Table.iterrows()]
    num_telescopes = f.root.Tel_Table.shape[0]
    # All telescope images have the same shape
    image_shape = f.root.E0._f_get_child(telescope_ids[0]).shape[1:]
    f.close()
    metadata = {
            'num_training_events': num_training_events,
            'num_validation_events': num_validation_events,
            'telescope_ids': telescope_ids,
            'num_telescopes': num_telescopes,
            'image_shape': image_shape,
            'num_auxiliary_inputs': 2,
            'num_gamma_hadron_classes': 2
            }
    return metadata

# Parse configuration file
config = configparser.ConfigParser()
try:
    config_filename = sys.argv[1]
except IndexError:
    sys.exit("Usage: train.py config_file")
config.read(config_filename)

# Load options related to loading the data
data_filename = config['Data']['Filename']
use_hdf5_format = config['Data'].getboolean('UseHDF5Format', False)

# Load options relating to processing the data
batch_size = config['Data Processing'].getint('BatchSize')
num_examples_per_training_epoch = config['Data Processing'].getint(
        'NumExamplesPerTrainingEpoch', 10000)
num_training_epochs_per_evaluation = config['Data Processing'].getint(
        'NumTrainingEpochsPerEvaluation', 1)
num_parallel_calls = config['Data Processing'].getint('NumParallelCalls', 1)

# Load options to specify the model
use_variable_input_model = config['Model'].getboolean('UseVariableInputModel', 
        False)
cnn_block = config['Model']['CNNBlock'].lower()
telescope_combination = config['Model']['TelescopeCombination'].lower()
network_head = config['Model']['NetworkHead'].lower()

# Load options for training hyperparameters
base_learning_rate = config['Training'].getfloat('BaseLearningRate')
batch_norm_decay = config['Training'].getfloat('BatchNormDecay', 0.95)

# Load options relating to logging and checkpointing
model_dir = config['Logging']['ModelDirectory']

# Log a copy of the configuration file
config_log_filename = time.strftime('%Y%m%d_%H%M%S_') + config_filename
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
shutil.copy(config_filename, os.path.join(model_dir, config_log_filename))

# Define data loading functions
if use_hdf5_format:
    load_data = load_HDF5_data
    load_auxiliary_data = load_HDF5_auxiliary_data
    load_metadata = load_HDF5_metadata
else:
    sys.exit("Error: No data format specified.")

# Define model
if use_variable_input_model:
    model = variable_input_model
else:
    sys.exit("Error: no valid model specified.")

# Define model hyperparameters
hyperparameters = {
        'cnn_block': cnn_block,
        'telescope_combination': telescope_combination,
        'network_head': network_head,
        'base_learning_rate': base_learning_rate,
        'batch_norm_decay': batch_norm_decay
        }

# Get information about the dataset
metadata = load_metadata(data_filename)

# Merge dictionaries for passing to the model function
params = {**hyperparameters, **metadata}

# Get the auxiliary input (same for every event)
auxiliary_data = load_auxiliary_data(data_filename)

# Create training and evaluation datasets
def load_training_data(index):
    return load_data(data_filename, index, metadata, mode='TRAIN')

def load_validation_data(index):
    return load_data(data_filename, index, metadata, mode='VALID')

training_dataset = tf.data.Dataset.range(metadata['num_training_events'])
training_dataset = training_dataset.map(lambda index: tuple(tf.py_func(
            load_training_data,
            [index], 
            [tf.float32, tf.int8, tf.int64])),
        num_parallel_calls=num_parallel_calls)
training_dataset = training_dataset.batch(batch_size)

validation_dataset = tf.data.Dataset.range(metadata['num_validation_events'])
validation_dataset = validation_dataset.map(lambda index: tuple(tf.py_func(
            load_validation_data,
            [index],
            [tf.float32, tf.int8, tf.int64])), 
        num_parallel_calls=num_parallel_calls)
validation_dataset = validation_dataset.batch(batch_size)

def input_fn(dataset, auxiliary_data, shuffle_buffer_size=None):
    # Get batches of data
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    iterator = dataset.make_one_shot_iterator()
    (telescope_data, telescope_triggers, 
            gamma_hadron_labels) = iterator.get_next()
    # Convert auxiliary data to tensors
    telescope_positions = tf.constant(auxiliary_data['telescope_positions'])
    features = {
            'telescope_data': telescope_data, 
            'telescope_triggers': telescope_triggers, 
            'telescope_positions': telescope_positions
            }
    labels = {
            'gamma_hadron_labels': gamma_hadron_labels
            }
    return features, labels

def model_fn(features, labels, mode, params, config):
    
    if (mode == tf.estimator.ModeKeys.TRAIN):
        is_training = True
    else:
        is_training = False
    
    loss, logits = model(features, labels, params, is_training)

    # Calculate metrics
    true_classes = tf.cast(labels['gamma_hadron_labels'], tf.int32)
    predicted_classes = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    training_accuracy = tf.reduce_mean(tf.cast(tf.equal(true_classes, 
        predicted_classes), tf.float32))
    predictions = {
            'classes': predicted_classes
            }
    
    # Scale the learning rate so batches with fewer triggered
    # telescopes don't have smaller gradients
    trigger_rate = tf.reduce_mean(tf.cast(features['telescope_triggers'], 
        tf.float32))
    # Avoid division by 0
    trigger_rate = tf.maximum(trigger_rate, 0.1)
    scaling_factor = tf.reciprocal(trigger_rate)
    scaled_learning_rate = tf.multiply(scaling_factor, 
            params['base_learning_rate'])
    
    # Define the train op
    optimizer = tf.train.AdamOptimizer(learning_rate=scaled_learning_rate)
    train_op = slim.learning.create_train_op(loss, optimizer)
    
    # Define the evaluation metrics
    eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(true_classes, predicted_classes),
            'auc': tf.metrics.auc(true_classes, predicted_classes)
            }
    
    return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

# Train and evaluate the model
print("Training and evaluating...")
print("Total number of training events: ", metadata['num_training_events'])
print("Total number of validation events: ", metadata['num_validation_events'])
print("Batch size: ", batch_size)
print("Training steps per epoch: ", np.ceil(metadata['num_training_events'] 
    / batch_size).astype(np.int32))

estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir,
        params=params)
while True:
    for _ in range(num_training_epochs_per_evaluation):
        estimator.train(lambda: input_fn(training_dataset, auxiliary_data, 
            shuffle_buffer_size=num_examples_per_training_epoch))
    estimator.evaluate(
            lambda: input_fn(training_dataset, auxiliary_data),
            name='training')
    estimator.evaluate(
            lambda: input_fn(validation_dataset, auxiliary_data),
            name='validation')
