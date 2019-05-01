import argparse
from collections import OrderedDict
import importlib
import logging
import math
import os
from pprint import pformat
import sys
import time

import pkg_resources
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import yaml

from dl1_data_handler.reader import DL1DataReader

# Disable Tensorflow info and warning messages (not error messages)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.logging.set_verbosity(tf.logging.WARN)

def setup_logging(config, log_dir, debug, log_to_file):

    # Log configuration to a text file in the log dir
    time_str = time.strftime('%Y%m%d_%H%M%S')
    config_filename = os.path.join(log_dir, time_str + '_config.yml')
    with open(config_filename, 'w') as outfile:
        ctlearn_version=pkg_resources.get_distribution("ctlearn").version
        outfile.write('# The training was performed using CTLearn version {}.\n'.format(ctlearn_version))
        yaml.dump(config, outfile, default_flow_style=False)

    # Set up logger
    logger = logging.getLogger()
    
    if debug: 
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    logger.handlers = [] # remove existing handlers from any previous runs
    if not log_to_file:
        handler = logging.StreamHandler()
    else:
        logging_filename = os.path.join(log_dir, time_str + '_logfile.log')
        handler = logging.FileHandler(logging_filename)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    logger.addHandler(handler)
    
    return logger

def run_model(config, mode="train", debug=False, log_to_file=False):

    # Load options relating to logging and checkpointing
    model_dir = config['Logging']['model_directory']
    # Create model directory if it doesn't exist already
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    # Set up logging, saving the config and optionally logging to a file
    logger = setup_logging(config, model_dir, debug, log_to_file)

    # Log the loaded configuration
    logger.debug(pformat(config))

    logger.info("Logging has been correctly set up")

    # Create params dictionary that will be passed to the model_fn
    params = {}
    
    # Load options to specify the model
    try:
        model_directory = config['Model']['model_directory']
        if model_directory is None:
            raise KeyError
    except KeyError:
        model_directory = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "default_models/"))
    sys.path.append(model_directory)
    model_module = importlib.import_module(config['Model']['model']['module'])
    model = getattr(model_module, config['Model']['model']['function'])
    model_type = config['Data'].get('Loading', {}).get('example_type', 'array')
    
    params['model'] = config['Model'].get('Model Parameters', {})
    params['model']['model_directory'] = model_directory

    # Parse file list
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
                         "Must be list or path to file".format(
                             config['Data']['file_list']))

    # Parse list of Transforms
    transforms = []
    for t in config['Data']['transforms']:
        if 'path' in t and t['path'] not in sys.path:
            sys.path.append(t['path'])
        module_name = t.get('module', 'dl1_data_handler.processor')
        module = importlib.import_module(module_name)
        transform = getattr(module, t['name'])(**t.get('args', {}))
        transforms.append(transform)
    config['Data']['transforms'] = transforms

    # Convert interpolation image shapes from lists to tuples, if present
    if 'interpolation_image_shape' in config['Data'].get('mapping_settings',
                                                         {}):
        config['Data']['mapping_settings']['interpolation_image_shape'] = {
                k: tuple(l) for k, l in config['Data']['mapping_settings']['interpolation_image_shape'].items()}

    # Create data reader
    reader = DL1DataReader(**config['Data'])
    params['example_description'] = reader.example_description

    # Define format for TensorFlow dataset
    config['Input']['output_names'] = [d['name'] for d
                                       in reader.example_description]
    config['Input']['output_dtypes'] = tuple(tf.as_dtype(d['dtype']) for d
                                             in reader.example_description)

    # Load either training or prediction options
    # and log information about the data set
    batch_size = config['Input'].get('batch_args', {}).get('batch_size', 1)
    logger.info("Batch size: {}".format(batch_size))
    
    if mode == 'train':

        params['training'] = config['Training']['Hyperparameters']
        params['training']['model_type'] = model_type
        num_validations = config['Training']['num_validations']
        train_forever = False if num_validations != 0 else True
        num_training_steps_per_validation = config['Training']['num_training_steps_per_validation']
        
        validation_split = config['Training']['validation_split']
        if not 0.0 < validation_split < 1.0:
            raise ValueError("Invalid validation split: {}. "
                             "Must be between 0.0 and 1.0".format(
                                 validation_split))
        num_validation_examples = math.ceil(validation_split * len(reader))
        num_training_examples = len(reader) - num_validation_examples
        indices = range(len(reader))
        training_indices = indices[:num_training_examples]
        validation_indices = indices[num_training_examples:]
        
        logger.info("Training and evaluating...")
        logger.info("Total number of training examples: {}".format(
                    num_training_examples))
        logger.info("Total number of validation examples: {}".format(
                    num_validation_examples))
        logger.info("Number of training steps per epoch: {}".format(
                    int(num_training_examples / batch_size)))
        logger.info("Number of training steps between validations: {}".format(
                    num_training_steps_per_validation))

    elif mode == 'predict':

        true_labels_given = config['Prediction']['true_labels_given']
        export_prediction_file = config['Prediction'].get('export_as_file',
                                                          False)
        if export_prediction_file:
            prediction_path = config['Prediction']['prediction_file_path']
        
        logger.info("Predicting...")
        logger.info("Total number of test examples: {}".format(len(reader)))
    
    # Load options for TensorFlow
    run_tfdbg = config.get('TensorFlow', {}).get('run_TFDBG', False)

    # Log the breakdown of examples by class
    group_by = config.get('Input', {}).get('label_names')
    logger.info("Number of examples by class: {}".format(
        reader.num_examples(group_by=group_by)))
    
    # Define input function for TF Estimator
    def input_fn(generator, output_names, output_dtypes, indices=None,
                 label_names=None, shuffle=False, shuffle_args=None,
                 batch=False, batch_args=None, prefetch=False,
                 prefetch_args=None):
        dataset = tf.data.Dataset.from_generator(generator, output_dtypes)#,
                                                 #args=(indices,))
        if shuffle:
            if shuffle_args is None: shuffle_args = {}
            dataset = dataset.shuffle(**shuffle_args)
        if batch:
            if batch_args is None: batch_args = {}
            dataset = dataset.batch(**batch_args)
        if prefetch:
            if prefetch_args is None: prefetch_args = {}
            dataset = dataset.prefetch(**prefetch_args)
    
        iterator = dataset.make_one_shot_iterator()

        # Return a batch of features and labels
        example = iterator.get_next()

        features, labels = {}, {}        
        for tensor, name in zip(example, output_names):
            dic = labels if name in label_names else features
            dic[name] = tensor

        return features, labels

    # Define model function with model, mode (train/predict),
    # metrics, optimizer, learning rate, etc.
    # to pass into TF Estimator
    def model_fn(features, labels, mode, params, config):
        
        training = True if mode == tf.estimator.ModeKeys.TRAIN else False
       
        logits = model(features, params['model'], training)
        
        # Collect predictions
        predictions = {}
        classifier_values = tf.nn.softmax(logits)
        predicted_classes = tf.cast(tf.argmax(classifier_values, axis=1),
                tf.int32, name="predicted_classes")
        predictions['predicted_class'] = predicted_classes
        for i in range(params['model']['num_classes']):
            class_name = params['model']['labels_to_class_names'][i]
            predictions[class_name] = classifier_values[:,i]
        
        # For predict mode, we're done
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions)

        training_params = params['training']

        # Compute class-weighted softmax-cross-entropy        
        true_classes = tf.cast(labels['gamma_hadron_label'], tf.int32,
                name="true_classes")

        # Get class weights
        if training_params['apply_class_weights']:
            class_weights = tf.constant(training_params['class_weights'],
                    dtype=tf.float32, name="class_weights") 
            weights = tf.gather(class_weights, true_classes, name="weights")
        else:
            weights = 1.0

        onehot_labels = tf.one_hot(indices=true_classes,
                depth=training_params['num_classes'])

        # compute cross-entropy loss
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, 
            logits=logits,weights=weights)
    
        # add regularization loss
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([loss] + regularization_losses, name="loss")

        # Compute accuracy
        training_accuracy = tf.reduce_mean(tf.cast(tf.equal(true_classes, 
            predicted_classes), tf.float32),name="training_accuracy")
        tf.summary.scalar("accuracy", training_accuracy)

        # Scale the learning rate so batches with fewer triggered
        # telescopes don't have smaller gradients
        # Only apply learning rate scaling for array-level models
        if (training_params['scale_learning_rate'] and
                model_type == 'array'):
            trigger_rate = tf.reduce_mean(tf.cast(
                features['telescope_triggers'], tf.float32),
                name="trigger_rate")
            trigger_rate = tf.maximum(trigger_rate, 0.1) # Avoid division by 0
            scaling_factor = tf.reciprocal(trigger_rate, name="scaling_factor")
            learning_rate = tf.multiply(scaling_factor, 
                training_params['base_learning_rate'], name="learning_rate")
        else:
            learning_rate = training_params['base_learning_rate']
        
        # Select optimizer with appropriate arguments

        # Dict of optimizer_name: (optimizer_fn, optimizer_args)
        optimizers = {
                'Adadelta': (tf.train.AdadeltaOptimizer,
                    dict(learning_rate=learning_rate)),
                'Adam': (tf.train.AdamOptimizer,
                    dict(learning_rate=learning_rate,
                        epsilon=training_params['adam_epsilon'])),
                'RMSProp': (tf.train.RMSPropOptimizer,
                    dict(learning_rate=learning_rate)),
                'SGD': (tf.train.GradientDescentOptimizer,
                    dict(learning_rate=learning_rate))
                }

        optimizer_fn, optimizer_args = optimizers[training_params['optimizer']]
        optimizer = optimizer_fn(**optimizer_args)
    
        var_list = None
        if training_params['variables_to_train'] is not None:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                    training_params['variables_to_train'])
        
        # Define train op with update ops dependency for batch norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss,
                    global_step=tf.train.get_global_step(),
                    var_list=var_list)
        
        # Define the evaluation metrics
        eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(true_classes, 
                    predicted_classes),
                'auc': tf.metrics.auc(1 - true_classes, predictions['gamma'])
                }
        
        # add class-wise accuracies
        for i in range(training_params['num_classes']):
            weights = tf.cast(tf.equal(true_classes,tf.constant(i)),tf.int32)
            eval_metric_ops['accuracy_{}'.format(
                training_params['labels_to_class_names'][i])] = tf.metrics.accuracy(
                        true_classes, predicted_classes, weights=weights)

        return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops)

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

    if mode == 'train':

        # Train and evaluate the model
        num_validations_remaining = num_validations
        while train_forever or num_validations_remaining:
            estimator.train(
                    lambda: input_fn(reader.generator,
                        indices=training_indices, **config['Input']),
                    steps=num_training_steps_per_validation, hooks=hooks)
            estimator.evaluate(
                    lambda: input_fn(reader.generator,
                        indices=validation_indices, **config['Input']),
                    hooks=hooks, name='validation')
            if not train_forever:
                num_validations_remaining -= 1

    elif mode == 'predict':

        prediction_output = OrderedDict()

        # Generate predictions and add to output
        predictions = estimator.predict(
                lambda: input_fn(reader.generator, **config['Input']),
                hooks=hooks)
        for event in predictions:
            for key, value in event.items():
                if key in prediction_output:
                    prediction_output[key].append(value)
                else:
                    prediction_output[key] = [value]
        
        # Get true labels and add to prediction output if available
        if true_labels_given:
            features, labels = input_fn(reader, **config['Input'])
            with tf.Session() as sess:
                while True:
                    try:
                        batch_labels = sess.run(labels)
                        for label, batch_vals in batch_labels.items():
                            if label in prediction_output:
                                prediction_output[label].extend(batch_vals)
                            else:
                                prediction_output[label] = []
                    except tf.errors.OutOfRangeError:
                        break
        
        # Get event ids of each example and add to prediction output
        if data_format == 'HDF5':
            event_ids = ['run_number', 'event_number']
            if data_loader.example_type == 'single_tel':
                event_ids.append('tel_id')
            for event_id in event_ids:
                prediction_output[event_id] = []
            for example in data_loader.examples:
                for i, event_id in enumerate(event_ids):
                    prediction_output[event_id].append(example[i])
        else:
            raise ValueError("Invalid data format: {}".format(data_format))

        # Write predictions and other info given a dictionary of input, with
        # the key:value pairs of header name: list of the values for each event
        def write_predictions(file_handle, prediction_output):
            header = ",".join([key for key in prediction_output]) + '\n'
            file_handle.write(header)
            output_lists = [value for key, value in prediction_output.items()]
            for output_values in zip(*output_lists):
                row = ",".join('{}'.format(value) for value in output_values)
                row += '\n'
                file_handle.write(row)

        # Write predictions to a csv file
        if export_prediction_file:
            with open(prediction_path, 'w') as predict_file:
                write_predictions(predict_file, prediction_output)
        else:
            write_predictions(sys.stdout, prediction_output)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description=("Train/Predict with a CTLearn model."))
    parser.add_argument(
            '--mode',
            default="train",
            help="Mode to run in (train/predict)")
    parser.add_argument(
            'config_file',
            help="path to YAML configuration file with training options")
    parser.add_argument(
            '--debug',
            action='store_true',
            help="print debug/logger messages")
    parser.add_argument(
            '--log_to_file',
            action='store_true',
            help="log to a file in model directory instead of terminal")

    args = parser.parse_args()
   
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

    run_model(config, mode=args.mode, debug=args.debug, log_to_file=args.log_to_file)
