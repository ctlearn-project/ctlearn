import argparse
import logging
import configparser
import os
import shutil
import sys
import time

import tensorflow as tf
slim = tf.contrib.slim

# Disable Tensorflow info and warning messages (not error messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.WARN)

# Logger
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def train(config):
    # Load options related to loading the data
    data_filelist = config['Data']['FileList']
    data_format = config['Data']['Format'].lower()
    sort_telescopes_by_trigger = config['Data'].getboolean(
            'SortTelescopesByTrigger', False)

    # Load options relating to processing the data
    batch_size = config['Data Processing'].getint('BatchSize')
    num_batches_per_training_epoch = config['Data Processing'].getint(
            'NumBatchesPerTrainingEpoch', 10000)
    num_training_epochs_per_evaluation = config['Data Processing'].getint(
            'NumTrainingEpochsPerEvaluation', 1)
    num_parallel_calls = config['Data Processing'].getint('NumParallelCalls', 1)
    validation_split = config['Data Processing'].getfloat('ValidationSplit',0.1)
    num_batches_per_evaluation = config['Data Processing'].getint('NumBatchesPerEvaluation',1000)
    cut_condition = config['Data Processing']['Cut Condition']

    # Load options to specify the model
    model_type = config['Model']['ModelType'].lower()
    if model_type == 'variableinputmodel':
        from ctalearn.models.variable_input_model import (
                variable_input_model as model)
        cnn_block = config['Model']['CNNBlock'].lower()
        telescope_combination = config['Model']['TelescopeCombination'].lower()
        network_head = config['Model']['NetworkHead'].lower()
    elif model_type == 'cnnrnn':
        from ctalearn.models.cnn_rnn import cnn_rnn_model as model
        cnn_block = config['Model']['CNNBlock'].lower()
        telescope_combination = None
        network_head = None
    elif model_type == 'singletel':
        from ctalearn.models.single_tel import single_tel_model as model
        cnn_block = config['Model']['CNNBlock'].lower()
        telescope_combination = None
        network_head = None
    else:
        sys.exit("Error: no valid model specified.")

    # Load options related to pretrained weights
    pretrained_weights_file = config['Model']['PretrainedWeights']
    freeze_weights = config['Model'].getboolean('FreezeWeights',False)

    # Load options for training hyperparameters
    optimizer_type = config['Training']['Optimizer'].lower()
    base_learning_rate = config['Training'].getfloat('BaseLearningRate')
    batch_norm_decay = config['Training'].getfloat('BatchNormDecay', 0.95)
    clip_gradient_norm = config['Training'].getfloat('ClipGradientNorm', 0.)

    # Load options relating to logging and checkpointing
    model_dir = config['Logging']['ModelDirectory']

    # Log a copy of the configuration file
    config_log_filename = time.strftime('%Y%m%d_%H%M%S_') + config_filename
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    shutil.copy(config_full_path, os.path.join(model_dir, config_log_filename))

    # Read data file list
    data_files = []
    with open(data_filelist) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != "#":
                data_files.append(line)

    # Define data loading functions
    if data_format == 'hdf5':
        if model_type == 'singletel':
           from ctalearn.data import load_HDF5_data_single_tel
            # For single telescope data, data types correspond to:
            # [telescope_data, gamma_hadron_label]
            data_types = [tf.float32, tf.int64]
        else:
            # Auxiliary data for array-level classification (telescope positions)
            from ctalearn.data import (
                load_HDF5_auxiliary_data as load_auxiliary_data)
            from ctalearn.data import load_HDF5_data
            # For array-level data, data types correspond to:
            # [telescope_data, telescope_triggers, telescope_positions,
            #  gamma_hadron_label]
            data_types = [tf.float32, tf.int8, tf.float32, tf.int64]

        # Both single-tel and array-level HDF5 data use the same metadata
        # and generator function
        from ctalearn.data import load_HDF5_metadata as load_metadata
        from ctalearn.data import HDF5_gen_fn as generator
    else:
        raise ValueError("Invalid data format: {}".format(data_format))

    # Define model hyperparameters
    hyperparameters = {
            'cnn_block': cnn_block,
            'telescope_combination': telescope_combination,
            'network_head': network_head,
            'base_learning_rate': base_learning_rate,
            'batch_norm_decay': batch_norm_decay,
            'clip_gradient_norm': clip_gradient_norm,
            'pretrained_weights': pretrained_weights_file,
            'freeze_weights': freeze_weights
            }
 
    # Get metadata information about the dataset
    metadata = load_metadata(data_files)

    # Merge dictionaries for passing to the model function
    params = {**hyperparameters, **metadata}

    # Set load data function and input_fn (for TF estimator) for single tel and array-level models
    if model_type == 'singletel':

        def load_data(filename,index):
            return load_HDF5_data_single_tel(filename, index, metadata)

        def input_fn(dataset):
            # Get batches of data
            iterator = dataset.make_one_shot_iterator()
            (telescope_data, gamma_hadron_labels) = iterator.get_next()
            features = {
                    'telescope_data': telescope_data 
                    }
            labels = {
                    'gamma_hadron_labels': gamma_hadron_labels
                    }
            return features, labels
    else: 
        # For array-level methods, get auxiliary data (telescope positions + other) in dict
        auxiliary_data = load_auxiliary_data(data_files)

        # Flatten auxiliary input dict into a list
        auxiliary_data_flattened = []
        for i in auxiliary_data.values():
            for j in i.values():
                auxiliary_data_flattened += j

        def load_data(filename,index):
            return load_HDF5_data(filename, index, auxiliary_data_flattened, metadata,sort_telescopes_by_trigger=sort_telescopes_by_trigger)

        def input_fn(dataset):
            # Get batches of data
            iterator = dataset.make_one_shot_iterator()
            (telescope_data, telescope_triggers, telescope_positions,
                    gamma_hadron_labels) = iterator.get_next()
            features = {
                    'telescope_data': telescope_data, 
                    'telescope_triggers': telescope_triggers, 
                    'telescope_positions': telescope_positions,
                    }
            labels = {
                    'gamma_hadron_labels': gamma_hadron_labels,
                    }
            return features, labels

    

    logger.info("{} data files read.".format(len(data_files)))
    logger.info("Telescopes in data:")
    for tel_type in metadata['telescope_ids']:
        logger.info(tel_type + ": "+'[%s]' % ', '.join(map(str,metadata['telescope_ids'][tel_type]))) 

    # Print metadata info on dataset
    num_examples_by_label = {}
    for i,num_examples in enumerate(num_examples_by_file):
        particle_id = metadata['particle_id_by_file'][i]
        if particle_id not in num_examples_by_label:
            num_examples_by_label[particle_id] = 0
        num_examples_by_label[particle_id] += num_examples

    num_examples = sum(num_examples_by_label.values())

    logger.info("{} total examples.".format(num_examples))
    logger.info("Num examples by label:")
    for label in num_examples_by_label:
        logger.info("{}: {} ({}%)".format(label,num_examples_by_label[label], 100 * float(num_examples_by_label[label])/num_examples))

    if cut_condition is not None:
        logger.info("Cut condition: {}".format(cut_condition))
    else:
        logger.info("No cuts applied.")

    indices_by_file = apply_cuts(data_files,cut_condition,model_type)

    # Log info on cuts
    num_passing_examples_by_label = {}
    for i,indices in enumerate(indices):
        num_passing_examples = len(indices)
        particle_id = metadata['particle_id_by_file'][i]
        if particle_id not in num_examples_by_label:
            num_passing_examples_by_label[particle_id] = 0
        num_passing_examples_by_label[particle_id] += num_passing_examples

    num_passing_examples = sum(num_passing_examples_by_label.values())
    num_validation_examples = int(validation_split * num_passing_examples)
    num_training_examples = num_passing_examples - num_validation_examples

    logger.info("{} total examples passing cuts.".format(num_passing_examples))
    logger.info("Num examples by label:")
    for label in num_examples_by_label:
        logger.info("{}: {} ({}%)".format(label,num_examples_by_label[label], 100 * float(num_examples_by_label[label])/num_examples))
 
    # Set generator function to create dataset of elements (filename,index)
    def gen_fn():
        return generator(data_files,indices_by_file)

    dataset = tf.data.Dataset.from_generator(gen_fn,(tf.string, tf.int64))
    dataset = dataset.shuffle(num_examples)
    dataset = dataset.map(lambda filename, index: tuple(tf.py_func(load_data,[filename, index], data_types)),num_parallel_calls=num_parallel_calls)

    training_dataset = dataset.skip(num_validation_examples)
    validation_dataset = dataset.take(num_validation_examples)

    training_dataset = training_dataset.batch(batch_size)
    validation_dataset = validation_dataset.batch(batch_size)

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
 
        # Only apply learning rate scaling for array-level models (not single tel)
        if model_type == 'singletel':
            learning_rate = params['base_learning_rate']
        else:
            # Scale the learning rate so batches with fewer triggered
            # telescopes don't have smaller gradients
            trigger_rate = tf.reduce_mean(tf.cast(features['telescope_triggers'], 
                tf.float32))
            # Avoid division by 0
            trigger_rate = tf.maximum(trigger_rate, 0.1)
            scaling_factor = tf.reciprocal(trigger_rate)
            learning_rate = tf.multiply(scaling_factor, 
                    params['base_learning_rate'])

        # Only apply gradient cliping if the model is CNNRNN
        if model_type == 'cnnrnn':
            clip_gradient_norm = params['clip_gradient_norm']
        else:
            clip_gradient_norm = 0

        # Define the train op
        if optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_type == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif optimizer_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError("Invalid optimizer type: {}".format(optimizer_type))

        train_op = slim.learning.create_train_op(loss, optimizer,
                clip_gradient_norm=clip_gradient_norm)
        
        # Define the evaluation metrics
        eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(true_classes, 
                    predicted_classes),
                'auc': tf.metrics.auc(true_classes, predicted_classes),
                }
        
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops)

    # Train and evaluate the model
    logger.info("Training and evaluating...")
    logger.info("Total number of training events: {}".format(num_training_examples))
    logger.info("Total number of validation events: {}".format(num_validation_examples))
    estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir, 
            params=params)
    while True:
        for _ in range(num_training_epochs_per_evaluation):
            estimator.train(lambda: input_fn(training_dataset), steps=num_batches_per_training_epoch)
        estimator.evaluate(
                lambda: input_fn(training_dataset), steps=num_batches_per_evaluation, name='training')
        estimator.evaluate(
                lambda: input_fn(validation_dataset), steps=num_batches_per_evaluation, name='validation')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description=("Train a ctalearn model."))
    parser.add_argument(
    'config_file',
    help='configuration file containing training options (to be parsed with configparser)')
    parser.add_argument(
    "--debug",
    help="print debug/logger messages",
    action="store_true")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Parse configuration file
    config = configparser.ConfigParser(allow_no_value=True)
    try:
        config_full_path = os.path.abspath(args.config_file)
        config_path, config_filename = os.path.split(config_full_path)
    except IndexError:
        raise ValueError("Invalid config file: {}".format(config_full_path))
    config.read(config_full_path)

    train(config)