import sys
import os
import argparse

# Disable info and warning messages (not error messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
slim = tf.contrib.slim
import tables
import numpy as np

# Add parent directory to pythonpath to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from models.variable_input_model import variable_input_model

NUM_PARALLEL_CALLS = 12
TRAINING_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 64

SHUFFLE_BUFFER_SIZE = 10000
MAX_STEPS = 10000

BATCH_NORM_DECAY = 0.95

def train(model, data_file, epochs, image_summary, embedding):

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
                # Telescope did not trigger. Its outputs will be
                # dropped out, so input is arbitrary. Use an empty
                # array for efficiency.
                telescope_images.append(np.empty(metadata['image_shape']))
            else:
                telescope_table = f.root.E0._f_get_child(telescope_id)
                telescope_images.append(telescope_table[image_index])
        telescope_images = np.stack(telescope_images).astype(np.float32)
        
        # Get binary values indicating whether each telescope triggered
        telescope_triggers = np.array([0 if i < 0 else 1 for i 
            in image_indices], dtype=np.int8)
        
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
            'telescope_positions': np.array(telescope_positions, 
                dtype=np.float32)
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

    # TODO: rename this argument
    model_dir = args.logdir

    # Define data loading functions
    load_data = load_HDF5_data
    load_auxiliary_data = load_HDF5_auxiliary_data
    load_metadata = load_HDF5_metadata

    # Get information about the dataset
    metadata = load_metadata(data_file)
    
    # Define model hyperparameters
    hyperparameters = {
            'base_learning_rate': args.lr,
            'batch_norm_decay': BATCH_NORM_DECAY
            }
    
    # Merge dictionaries for passing to the model function
    params = {**metadata, **hyperparameters}
    
    # Get the auxiliary input (same for every event)
    auxiliary_data = load_auxiliary_data(data_file)
   
    # Create training and evaluation datasets
    def load_training_data(index):
        return load_data(data_file, index, metadata, mode='TRAIN')
    
    def load_validation_data(index):
        return load_data(data_file, index, metadata, mode='VALID')

    training_dataset = tf.data.Dataset.range(metadata['num_training_events'])
    training_dataset = training_dataset.map(lambda index: tuple(tf.py_func(
                load_training_data,
                [index], 
                [tf.float32, tf.int8, tf.int64])),
            num_parallel_calls=NUM_PARALLEL_CALLS)
    training_dataset = training_dataset.batch(TRAINING_BATCH_SIZE)

    validation_dataset = tf.data.Dataset.range(
            metadata['num_validation_events'])
    validation_dataset = validation_dataset.map(lambda index: tuple(tf.py_func(
                load_validation_data,
                [index],
                [tf.float32, tf.int8, tf.int64])), 
            num_parallel_calls=NUM_PARALLEL_CALLS)
    validation_dataset = validation_dataset.batch(VALIDATION_BATCH_SIZE)

    def input_fn(dataset, auxiliary_data, shuffle_buffer_size=None):
        # Get batches of data
        if shuffle_buffer_size:
            dataset = dataset.shuffle(shuffle_buffer_size)
        iterator = dataset.make_one_shot_iterator()
        (telescope_data, telescope_triggers, 
                gamma_hadron_labels) = iterator.get_next()
        # Convert auxiliary data to tensors
        telescope_positions = tf.constant(
                auxiliary_data['telescope_positions'])
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
        
        # Define the summaries
        #tf.summary.scalar('training_accuracy', training_accuracy)
        tf.summary.scalar('scaled_learning_rate', scaled_learning_rate)
        tf.summary.merge_all()
        # Define the evaluation metrics
        eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(true_classes, 
                    predicted_classes),
                'auc': tf.metrics.auc(true_classes,
                    predicted_classes)
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
    print("Total number of validation events: ", 
            metadata['num_validation_events'])
    print("Training batch size: ", TRAINING_BATCH_SIZE)
    print("Validation batch size: ", VALIDATION_BATCH_SIZE)
    print("Training steps per epoch: ", np.ceil(metadata['num_train_events'] 
        / TRAINING_BATCH_SIZE).astype(np.int32))
    
    estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir,
            params=params)
    while True:
        estimator.train(lambda: input_fn(training_dataset, auxiliary_data, 
            shuffle_buffer_size=SHUFFLE_BUFFER_SIZE))
        estimator.evaluate(
                lambda: input_fn(training_dataset, auxiliary_data),
                name='training')
        estimator.evaluate(
                lambda: input_fn(validation_dataset, auxiliary_data),
                name='validation')

if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Trains on an hdf5 file.')
    parser.add_argument('h5_file', help='path to h5 file containing data')
    parser.add_argument('--optimizer',default='adam')
    parser.add_argument('--epochs',default=10000,type=int)
    parser.add_argument('--logdir',default='/data0/logs/variable_input_model_1')
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--label_col_name',default='gamma_hadron_label')
    parser.add_argument('--checkpoint_basename',default='custom_multi_input.ckpt')
    parser.add_argument('--embedding', action='store_true')
    parser.add_argument('--no_val',action='store_true')
    parser.add_argument('--image_summary',action='store_true')
    args = parser.parse_args()

    train(variable_input_model,args.h5_file,args.epochs,args.image_summary,args.embedding)
