import argparse
import logging
import configparser
import os
import shutil
import sys
import time

import tensorflow as tf

import ctalearn.data

# Disable Tensorflow info and warning messages (not error messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.WARN)


def train(config):
    # Load options related to loading the data
    data_files = []
    with open(config['Data']['DataFilesList']) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != "#":
                data_files.append(line)
    data_format = config['Data']['Format'].lower()
    sort_telescopes_by_trigger = config['Data'].getboolean(
        'SortTelescopesByTrigger', False)

    # Load options relating to processing the data
    batch_size = config['Data Processing'].getint('BatchSize')
    num_training_steps_per_validation = config['Data Processing'].getint(
        'NumTrainingStepsPerValidation', 1000)
    prefetch_buffer_size = (config['Data Processing'].getint(
        'PrefetchBufferSize') if 
        config['Data Processing']['PrefetchBufferSize'] else None)
    num_parallel_calls = config['Data Processing'].getint(
        'NumParallelCalls', 12)
    validation_split = config['Data Processing'].getfloat(
        'ValidationSplit',0.1)
    cut_condition = config['Data Processing']['CutCondition'] if config['Data Processing']['CutCondition'] else None

    # Load options to specify the model
    model_type = config['Model']['ModelType'].lower()
    if model_type == 'variableinputmodel':
        from ctalearn.models.variable_input_model import (
                variable_input_model as model)
        cnn_block = config['Model']['CNNBlock'].lower()
        network_head = config['Model']['NetworkHead'].lower()
    elif model_type == 'cnnrnn':
        from ctalearn.models.cnn_rnn import cnn_rnn_model as model
        cnn_block = config['Model']['CNNBlock'].lower()
        network_head = None
    elif model_type == 'singletel':
        from ctalearn.models.single_tel import single_tel_model as model
        cnn_block = config['Model']['CNNBlock'].lower()
        network_head = None
    else:
        raise ValueError("Invalid model type: {}".format(model_type))

    # Load options related to pretrained weights
    pretrained_weights_file = config['Model']['PretrainedWeights'] if config['Model']['PretrainedWeights'] else None
    freeze_weights = config['Model'].getboolean('FreezeWeights',False)

    # Load options for training hyperparameters
    optimizer_type = config['Training']['Optimizer'].lower()
    base_learning_rate = config['Training'].getfloat('BaseLearningRate')
    scale_learning_rate = config['Training'].getboolean('ScaleLearningRate',False)
    batch_norm_decay = config['Training'].getfloat('BatchNormDecay', 0.95)
    clip_gradient_norm = config['Training'].getfloat('ClipGradientNorm', 0.)
    
    # Load options relating to logging and checkpointing
    model_dir = config['Logging']['ModelDirectory']

    # Load options related to debugging
    run_tfdbg = config['Debug'].getboolean('RunTFDBG',False)

    # Log a copy of the configuration file
    config_log_filename = time.strftime('%Y%m%d_%H%M%S_') + config_filename
    shutil.copy(config_full_path, os.path.join(model_dir, config_log_filename))

    # Define model hyperparameters
    hyperparameters = {
            'cnn_block': cnn_block,
            'network_head': network_head,
            'base_learning_rate': base_learning_rate,
            'batch_norm_decay': batch_norm_decay,
            'clip_gradient_norm': clip_gradient_norm,
            'pretrained_weights': pretrained_weights_file,
            'freeze_weights': freeze_weights
            }
 
    # Define data loading functions
    if data_format == 'hdf5':

        # Load metadata from HDF5 files
        metadata = ctalearn.data.load_metadata_HDF5(data_files)
 
        if model_type == 'singletel':
            # NOTE: Single tel mode currently hardocoded to read MSTS images
            # only 
            def load_data(filename,index):
                return ctalearn.data.load_data_single_tel_HDF5(
                        filename,
                        'MSTS',
                        index,
                        metadata)

            # Output datatypes of load_data (required by tf.py_func)
            # For single telescope data, data types correspond to:
            # [telescope_data, gamma_hadron_label]
            data_types = [tf.float32, tf.int64]

        else:
            # For array-level methods, get a dict of auxiliary data (telescope
            # positions and any other data)
            auxiliary_data = ctalearn.load_auxiliary_data_HDF5(data_files)

            def load_data(filename,index):
                return ctalearn.data.load_data_eventwise_HDF5(
                        filename, 
                        index, 
                        auxiliary_data, 
                        metadata,
                        sort_telescopes_by_trigger=sort_telescopes_by_trigger)

            # Output datatypes of load_data (required by tf.py_func)
            # For array-level data, data types correspond to:
            # [telescope_data, telescope_triggers, telescope_positions,
            #  gamma_hadron_label]
            data_types = [tf.float32, tf.int8, tf.float32, tf.int64]

        # Define format for Tensorflow dataset
        # Build dataset from generator returning (HDF5_filename, index) pairs
        # and a load_data function which maps (HDF5_filename, index) pairs
        # to full training examples (images and labels)
        generator_output_types = (tf.string, tf.int64)
        map_func = lambda filename, index: tuple(tf.py_func(load_data,
            [filename, index], data_types))
        
        # Get data generators returning (filename,index) pairs from data files 
        # by applying cuts and splitting into training and validation
        training_generator, validation_generator = (
                ctalearn.data.get_data_generators_HDF5(data_files,
                    cut_condition, model_type, validation_split))

    else:
        raise ValueError("Invalid data format: {}".format(data_format))

    # Define input function for TF Estimator
    def input_fn(generator, repeat=False): 
        # NOTE: Dataset.from_generator takes a callable (i.e. a generator
        # function / function returning a generator) not a python generator
        # object. To get the generator object from the function (i.e. to
        # measure its length), the function must be called (i.e. generator())
        dataset = tf.data.Dataset.from_generator(generator,
                generator_output_types).shuffle(len(list(generator())))
        dataset = dataset.map(map_func, num_parallel_calls=num_parallel_calls)
        if repeat:
            dataset = dataset.repeat() 
        dataset = dataset.batch(batch_size)
        if prefetch_buffer_size:
            dataset = dataset.prefetch(prefetch_buffer_size)
    
        iterator = dataset.make_one_shot_iterator()

        # For single tel, return a batch of images and labels
        if model_type == 'singletel':
            (telescope_data, gamma_hadron_label) = iterator.get_next()
            features = {
                    'telescope_data': telescope_data
                    }
            labels = {
                    'gamma_hadron_label': gamma_hadron_label,
                    }
        # For array-level, return a batch of images, triggers, telescope
        # positions, and labels
        else:
            (telescope_data, telescope_triggers, telescope_positions, gamma_hadron_label) = iterator.get_next()
            features = {
                    'telescope_data': telescope_data, 
                    'telescope_triggers': telescope_triggers, 
                    'telescope_positions': telescope_positions,
                    }
            labels = {
                    'gamma_hadron_label': gamma_hadron_label,
                    }
            
        return features, labels

    # Merge dictionaries for passing to the model function
    params = {**hyperparameters, **metadata}

    # Define model function with model, mode (train/predict),
    # metrics, optimizer, learning rate, etc.
    # to pass into TF Estimator
    def model_fn(features, labels, mode, params, config):
        
        is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False
       
        loss, logits = model(features, labels, params, is_training)

        # Calculate metrics
        true_classes = tf.cast(labels['gamma_hadron_label'], tf.int32)
        predicted_classes = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        training_accuracy = tf.reduce_mean(tf.cast(tf.equal(true_classes, 
            predicted_classes), tf.float32))
        predictions = {
                'classes': predicted_classes
                }

        tf.summary.scalar("accuracy",training_accuracy)

        # Scale the learning rate so batches with fewer triggered
        # telescopes don't have smaller gradients
        if scale_learning_rate:
            # Only apply learning rate scaling for array-level models (not single tel)
            if model_type == 'singletel': 
                raise ValueError("Learning rate scaling not valid for single tel model")
            
            trigger_rate = tf.reduce_mean(tf.cast(features['telescope_triggers'], tf.float32))
            trigger_rate = tf.maximum(trigger_rate, 0.1) # Avoid division by 0
            scaling_factor = tf.reciprocal(trigger_rate)
            learning_rate = tf.multiply(scaling_factor, 
                params['base_learning_rate'])
        else:
            learning_rate = params['base_learning_rate']
        
        # Select optimizer and set learning rate
        if optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=0.1)
        elif optimizer_type == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif optimizer_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError("Invalid optimizer choice: {}".format(optimizer_type))

        train_op = tf.contrib.slim.learning.create_train_op(loss, optimizer,
                clip_gradient_norm=params['clip_gradient_norm'])
        
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

    # Log information on number of training and validation events
    num_training_examples = len(list(training_generator()))
    num_validation_examples = len(list(validation_generator()))
    
    logger.info("Training and evaluating...")
    logger.info("Total number of training events: {}".format(num_training_examples))
    logger.info("Total number of validation events: {}".format(num_validation_examples))
    logger.info("Batch size: {}".format(batch_size))

    logger.info("Number of training steps per epoch: {}".format(int(num_training_examples/batch_size)))
    logger.info("Number of training steps per validation: {}".format(num_training_steps_per_validation))
 
    estimator = tf.estimator.Estimator(
            model_fn, 
            model_dir=model_dir, 
            params=params)

    # Set monitors and hooks
    monitors_and_hooks = [
            tf.contrib.learn.monitors.ValidationMonitor(
                input_fn= lambda: input_fn(validation_generator,repeat=False),
                every_n_steps=num_training_steps_per_validation,
                name="validation")]

    hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(monitors_and_hooks, estimator)

    # Activate Tensorflow debugger if appropriate option set
    if run_tfdbg:
        hooks.append(tf.python.debug.LocalCLIDebugHook())

    # Train and evaluate model
    estimator.train(lambda: input_fn(training_generator,repeat=True), steps=None, hooks=hooks)

    """
    while True:
        for _ in range(num_training_epochs_per_evaluation):
            estimator.train(lambda: input_fn(training_generator,shuffle=True), steps=num_batches_per_training_epoch, hooks=hooks)
        estimator.evaluate(
                lambda: input_fn(training_generator,shuffle=True), steps=num_batches_per_train_eval,hooks=hooks, name='training')
        estimator.evaluate(
                lambda: input_fn(validation_generator,shuffle=True), steps=num_batches_per_val_eval,hooks=hooks,  name='validation')
    """

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
    parser.add_argument(
        "--log_to_file",
        help="name of log file to write to. If not provided, will write to terminal",
        action="store_true")

    args = parser.parse_args()
   
    # Parse configuration file
    config = configparser.ConfigParser(allow_no_value=True)
    try:
        config_full_path = os.path.abspath(args.config_file)
        config_path, config_filename = os.path.split(config_full_path)
    except IndexError:
        raise ValueError("Invalid config file: {}".format(config_full_path))
    config.read(config_full_path)
    
    # Logger setup
    logger = logging.getLogger()
    if args.debug: logger.setLevel(logging.DEBUG)

    # Create model directory if it doesn't exist already
    model_dir = config['Logging']['ModelDirectory']
    if not os.path.exists(model_dir): os.makedirs(model_dir)
 
    handler = logging.FileHandler(os.path.join(model_dir,time.strftime('%Y%m%d_%H%M%S_') + 'logfile.log')) if args.log_to_file else logging.StreamHandler() 
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    
    logger.addHandler(handler)
 
    train(config)
