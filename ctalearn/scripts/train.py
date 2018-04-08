import argparse
import logging
import configparser
import os
import shutil
import sys
import time

import tensorflow as tf
from tensorflow.python import debug as tf_debug

import ctalearn.data
from ctalearn.models.variable_input_model import variable_input_model
from ctalearn.models.cnn_rnn import cnn_rnn_model
from ctalearn.models.single_tel import single_tel_model

# Disable Tensorflow info and warning messages (not error messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.WARN)


def train(config):
    # Load options related to the data format and location
    data_format = config['Data Format']['Format'].lower()
    data_files = []
    with open(config['Data Format']['DataFilesList']) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != "#":
                data_files.append(line)

    # Load options related to data input
    batch_size = config['Data Input'].getint('BatchSize')
    num_parallel_calls = config['Data Input'].getint(
        'NumParallelCalls', 1)
    prefetch = config['Data Input'].getboolean('Prefetch', True)
    prefetch_buffer_size = config['Data Input'].getint('PrefetchBufferSize', 10)
    shuffle = config['Data Input'].getboolean('Shuffle', True)
    shuffle_buffer_size = config['Data Input'].getint('ShuffleBufferSize',10000)

    # Load options related to data processing
    validation_split = config['Data Processing'].getfloat(
        'ValidationSplit',0.1)
    cut_condition = config['Data Processing'].get('CutCondition', '')
    min_num_tels = config['Data Processing'].getint('MinNumTels', 1)
    sort_telescopes_by_trigger = config['Data Processing'].getboolean(
        'SortTelescopesByTrigger', False)
    use_telescope_positions = config['Data Processing'].getboolean(
            'UseTelescopePositions', True)
    crop_images = config['Data Processing'].getboolean('CropImages', False)
    log_normalize_charge = config['Data Processing'].getboolean('LogNormalizeCharge', False)
    image_cleaning_method = config['Data Processing'].get(
            'ImageCleaningMethod', 'None').lower()
    return_cleaned_images = config['Data Processing'].getboolean(
            'ReturnCleanedImages', False)
    bounding_box_size = config['Data Processing'].getint(
        'BoundingBoxSize', 48)
    picture_threshold = config['Data Processing'].getfloat(
        'PictureThreshold', 5.5)
    boundary_threshold = config['Data Processing'].getfloat(
        'BoundaryThreshold', 1.0)

    # Load options to specify the model
    model_type = config['Model']['ModelType'].lower()
    if model_type == 'variableinputmodel':
        model = variable_input_model
        cnn_block = config['Model']['CNNBlock'].lower()
        network_head = config['Model']['NetworkHead'].lower()
    elif model_type == 'cnnrnn':
        model = cnn_rnn_model
        cnn_block = config['Model']['CNNBlock'].lower()
        network_head = None
    elif model_type == 'singletel':
        model = single_tel_model
        cnn_block = config['Model']['CNNBlock'].lower()
        network_head = None
    else:
        raise ValueError("Invalid model type: {}".format(model_type))

    # Load options related to pretrained weights
    pretrained_weights = config['Model'].get('PretrainedWeights', '')
    freeze_weights = config['Model'].getboolean('FreezeWeights', False)

    # Load options related to training hyperparameters
    optimizer_type = config['Training Hyperparameters']['Optimizer'].lower()
    base_learning_rate = config['Training Hyperparameters'].getfloat(
            'BaseLearningRate')
    scale_learning_rate = config['Training Hyperparameters'].getboolean(
            'ScaleLearningRate', False)
    batch_norm_decay = config['Training Hyperparameters'].getfloat(
            'BatchNormDecay', 0.95)
    clip_gradient_norm = config['Training Hyperparameters'].getfloat(
            'ClipGradientNorm', 0.)
    apply_class_weights = config['Training Hyperparameters'].getboolean(
            'ApplyClassWeights', False)

    # Load options related to training settings
    num_epochs = config['Training Settings'].getint('NumEpochs', 0)
    if num_epochs < 0:
        raise ValueError("NumEpochs must be positive or 0: invalid value {}".format(num_epochs))
    train_forever = False if num_epochs else True
    num_training_steps_per_validation = config['Training Settings'].getint(
        'NumTrainingStepsPerValidation', 1000)
    
    # Load options relating to logging and checkpointing
    model_dir = config['Logging']['ModelDirectory']

    # Load options related to debugging
    run_tfdbg = config['Debug'].getboolean('RunTFDBG', False)

    # Log a copy of the configuration file
    config_log_filename = time.strftime('%Y%m%d_%H%M%S_') + config_filename
    shutil.copy(config_full_path, os.path.join(model_dir, config_log_filename))

    # Define data input settings
    data_input_settings = {
            'batch_size': batch_size,
            'prefetch': prefetch,
            'prefetch_buffer_size': prefetch_buffer_size,
            'map': False,
            'num_parallel_calls': num_parallel_calls,
            'shuffle': shuffle,
            'shuffle_buffer_size': shuffle_buffer_size
            }

    # Define data processing settings
    data_processing_settings = {
            'validation_split': validation_split,
            'min_num_tels': min_num_tels,
            'cut_condition': cut_condition,
            'sort_telescopes_by_trigger': sort_telescopes_by_trigger,
            'use_telescope_positions': use_telescope_positions,
            'crop_images': crop_images,
            'log_normalize_charge': log_normalize_charge,
            'image_cleaning_method': image_cleaning_method,
            'return_cleaned_images': return_cleaned_images,
            'picture_threshold': picture_threshold,
            'boundary_threshold': boundary_threshold,
            'bounding_box_size': bounding_box_size,
            'num_shower_coordinates': 2,
            'model_type': model_type, # for applying cuts
            'chosen_telescope_types': ['MSTS'] # hardcode using SCT images only
            }
    
    # Define model hyperparameters
    hyperparameters = {
            'cnn_block': cnn_block,
            'network_head': network_head,
            'base_learning_rate': base_learning_rate,
            'batch_norm_decay': batch_norm_decay,
            'clip_gradient_norm': clip_gradient_norm,
            'pretrained_weights': pretrained_weights,
            'freeze_weights': freeze_weights,
            'apply_class_weights': apply_class_weights
            }

    # Define data loading functions
    if data_format == 'hdf5':

        # Load metadata from HDF5 files
        metadata = ctalearn.data.load_metadata_HDF5(data_files)

        # Calculate the post-processing image and telescope parameters that
        # depend on both the data processing and metadata, adding them to both
        # dictionaries
        ctalearn.data.add_processed_parameters(data_processing_settings,
                metadata)
 
        if model_type == 'singletel':
            def load_data(filename, index):
                return ctalearn.data.load_data_single_tel_HDF5(
                        filename,
                        index,
                        metadata,
                        data_processing_settings)

            # Output datatypes of load_data (required by tf.py_func)
            data_types = [tf.float32, tf.int64]
            output_names = ['telescope_data', 'gamma_hadron_label']
            outputs_are_label = [False, True]

        else:
            # For array-level methods, get a dict of auxiliary data (telescope
            # positions and any other data)
            auxiliary_data = ctalearn.data.load_auxiliary_data_HDF5(data_files)

            def load_data(filename,index):
                return ctalearn.data.load_data_eventwise_HDF5(
                        filename,
                        index,
                        auxiliary_data,
                        metadata,
                        data_processing_settings)

            # Output datatypes of load_data (required by tf.py_func)
            data_types = [tf.float32, tf.int8, tf.float32, tf.int64]
            output_names = ['telescope_data', 'telescope_triggers',
                    'telescope_aux_inputs', 'gamma_hadron_label']
            outputs_are_label = [False, False, False, True]

        # Define format for Tensorflow dataset
        # Build dataset from generator returning (HDF5_filename, index) pairs
        # and a load_data function which maps (HDF5_filename, index) pairs
        # to full training examples (images and labels)
        generator_output_types = (tf.string, tf.int64)
        map_func = lambda filename, index: tuple(tf.py_func(load_data,
            [filename, index], data_types))

        data_input_settings['generator_output_types'] = generator_output_types
        data_input_settings['map'] = True
        data_input_settings['map_func'] = map_func
        data_input_settings['output_names'] = output_names
        data_input_settings['outputs_are_label'] = outputs_are_label
        
        # Get data generators returning (filename,index) pairs from data files 
        # by applying cuts and splitting into training and validation
        training_generator, validation_generator = (
                ctalearn.data.get_data_generators_HDF5(data_files, metadata,
                    data_processing_settings))

    else:
        raise ValueError("Invalid data format: {}".format(data_format))

    # Define input function for TF Estimator
    def input_fn(generator, settings): 
        # NOTE: Dataset.from_generator takes a callable (i.e. a generator
        # function / function returning a generator) not a python generator
        # object. To get the generator object from the function (i.e. to
        # measure its length), the function must be called (i.e. generator())
        dataset = tf.data.Dataset.from_generator(generator,
                settings['generator_output_types'])
        if settings['shuffle']:
            dataset = dataset.shuffle(settings['shuffle_buffer_size'])
        if settings['map']:
            dataset = dataset.map(settings['map_func'],
                    num_parallel_calls=settings['num_parallel_calls'])
        dataset = dataset.batch(settings['batch_size'])
        if settings['prefetch']:
            dataset = dataset.prefetch(settings['prefetch_buffer_size'])
    
        iterator = dataset.make_one_shot_iterator()

        # Return a batch of features and labels. For example, for an
        # array-level network the features are images, triggers, and telescope
        # positions, and the labels are the gamma-hadron labels
        iterator_outputs = iterator.get_next()
        features = {}
        labels = {}
        for output, output_name, is_label in zip(
                iterator_outputs,
                settings['output_names'],
                settings['outputs_are_label']):
            if is_label:
                labels[output_name] = output
            else:
                features[output_name] = output

        return features, labels

    # Merge dictionaries for passing to the model function
    params = {**hyperparameters, **metadata}

    # Define model function with model, mode (train/predict),
    # metrics, optimizer, learning rate, etc.
    # to pass into TF Estimator
    def model_fn(features, labels, mode, params, config):
        
        is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False
       
        logits = model(features, labels, params, is_training)

        # Collect true labels and predictions
        true_classes = tf.cast(labels['gamma_hadron_label'], tf.int32, name="true_classes")
        predicted_classes = tf.cast(tf.argmax(logits, axis=1), tf.int32, name="predicted_classes")
        
        # Compute class-weighted softmax-cross-entropy

        # get class weights
        if params['apply_class_weights']:
            class_weights = tf.constant(params['class_weights'], dtype=tf.float32, name="class_weights") 
            weights = tf.gather(class_weights, true_classes, name="weights")
        else:
            weights = 1.0

        onehot_labels = tf.one_hot(indices=true_classes,depth=params['num_classes'])

        # compute cross-entropy loss
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, 
            logits=logits,weights=weights)
    
        # add regularization loss
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([loss] + regularization_losses, name="loss")

        # compute accuracy and predictions 
        training_accuracy = tf.reduce_mean(tf.cast(tf.equal(true_classes, 
            predicted_classes), tf.float32),name="training_accuracy")
        predictions = {
                'classes': predicted_classes
                }

        tf.summary.scalar("accuracy",training_accuracy)

        # Scale the learning rate so batches with fewer triggered
        # telescopes don't have smaller gradients
        # Only apply learning rate scaling for array-level models
        if scale_learning_rate and model_type != 'singletel':
            trigger_rate = tf.reduce_mean(tf.cast(features['telescope_triggers'], tf.float32), name="trigger_rate")
            trigger_rate = tf.maximum(trigger_rate, 0.1) # Avoid division by 0
            scaling_factor = tf.reciprocal(trigger_rate, name="scaling_factor")
            learning_rate = tf.multiply(scaling_factor, 
                params['base_learning_rate'], name="learning_rate")
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
    
        if params['freeze_weights']:
            vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "NetworkHead")
        else:
            vars_to_train = None

        train_op = tf.contrib.slim.learning.create_train_op(loss, optimizer,
                clip_gradient_norm=params['clip_gradient_norm'],
                variables_to_train=vars_to_train)
        
        # Define the evaluation metrics
        eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(true_classes, 
                    predicted_classes),
                'auc': tf.metrics.auc(true_classes, predicted_classes)
                }
        
        # add class-wise accuracies
        for i in range(params['num_classes']):
            weights = tf.cast(tf.equal(true_classes,tf.constant(i)),tf.int32)
            eval_metric_ops['accuracy_{}'.format(params['class_to_name'][i])] = tf.metrics.accuracy(true_classes,predicted_classes,weights=weights)

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

    hooks = None
    # Activate Tensorflow debugger if appropriate option set
    if run_tfdbg:
        if not isinstance(hooks, list):
            hooks = []
        hooks.append(tf_debug.LocalCLIDebugHook())
    
    num_epochs_remaining = num_epochs
    while train_forever or num_epochs_remaining:
        estimator.train(
                lambda: input_fn(training_generator, data_input_settings),
                steps=num_training_steps_per_validation, hooks=hooks)
        estimator.evaluate(
                lambda: input_fn(validation_generator, data_input_settings),
                hooks=hooks, name='validation')
        if not train_forever:
            num_epochs_remaining -= 1

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
