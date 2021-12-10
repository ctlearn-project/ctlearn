import argparse
import importlib
import logging
import math
from random import randint
import os
import glob
from pprint import pformat
import sys

import numpy as np
import yaml

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from dl1_data_handler.reader import DL1DataReaderSTAGE1, DL1DataReaderDL1DH
from ctlearn.default_models.basic import fc_head
from ctlearn.data_loader import *
from ctlearn.output_handler import *
from ctlearn.utils import *

# Disable Tensorflow info and warning messages (not error messages)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.logging.set_verbosity(tf.logging.WARN)

def run_model(config, mode="train", debug=False, log_to_file=False):

    # Load options relating to logging and checkpointing
    root_model_dir = model_dir = config['Logging']['model_directory']
    # Create model directory if it doesn't exist already
    if not os.path.exists(root_model_dir):
        if mode == 'predict':
            raise ValueError("Invalid model directory '{}'. "
            "Must be a path to an existing directory in the predict mode.".format(config['Logging']['model_directory']))
        os.makedirs(root_model_dir)

    random_seed = None
    if config['Logging'].get('add_seed', False):
        random_seed = config['Data']['seed']
        model_dir += "/seed_{}".format(random_seed)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

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

    params['model'] = {**config['Model'], **config.get('Model Parameters', {})}
    tasks = config['Model']['tasks']

    # Set up the DL1DataReaderSTAGE1
    config['Data'] = setup_DL1DataReader(config, mode)

    # Create data reader
    logger.info("Loading data:")
    logger.info("For a large dataset, this may take a while...")

    if config['Data_format'] == 'stage1':
        reader = DL1DataReaderSTAGE1(**config['Data'])
    elif config['Data_format'] == 'dl1dh':
        reader = DL1DataReaderDL1DH(**config['Data'])
    else:
        raise ValueError("Data format {} is not implemented in the DL1DH reader. Available data formats are 'stage1' and 'dl1dh'.".format(config['Data_format']))

    params['example_description'] = reader.example_description

    # Set up the TensorFlow dataset
    if 'Input' not in config:
        config['Input'] = {}

    config['Input'] = setup_TFdataset_format(config, params['example_description'], tasks)
    batch_size = config['Input'].get('batch_size', 1)

    # Load either training or prediction options
    # and log information about the data set
    indices = list(range(len(reader)))

    # Write the training configuration in the params dict
    params['training'] = config['Training']

    if mode in ['train', 'load_only']:

        # Write the training configuration in the params dict
        params['training'] = config['Training']

        validation_split = config['Training']['validation_split']
        if not 0.0 < validation_split < 1.0:
            raise ValueError("Invalid validation split: {}. "
                             "Must be between 0.0 and 1.0".format(
                                 validation_split))
        num_training_examples = math.floor((1 - validation_split) * len(reader))
        training_indices = indices[:num_training_examples]
        validation_indices = indices[num_training_examples:]
        logger.info("Number of epochs: {}".format(
            config['Training']['num_epochs']))
        logger.info("Number of training steps per epoch: {}".format(
            int(num_training_examples / batch_size)))

    if mode == 'load_only':

        log_examples(reader, indices, tasks, 'total dataset')
        log_examples(reader, training_indices, tasks, 'training')
        log_examples(reader, validation_indices, tasks, 'validation')
        # If only loading data, can end now that dataset logging is complete
        return

    if mode == 'train' and config['Training']['apply_class_weights']:
        num_class_examples = log_examples(reader, training_indices,
                                          tasks, 'training',
                                          group_by=['particletype'])
        try:
            class_labels = tasks['particletype']['class_names']
        except KeyError:
            raise ValueError("Applying class weights is supported"
                             " only for the particletype task.")
        class_weights = compute_class_weights(class_labels, num_class_examples)
        params['training']['class_weights'] = class_weights

    # Load options for TensorFlow
    run_tfdbg = config.get('TensorFlow', {}).get('run_TFDBG', False)

    # Define model function with model, mode (train/predict),
    # metrics, optimizer, learning rate, etc.
    # to pass into TF Estimator
    def model_fn(features, labels, mode, params):

        training = True if mode == tf.estimator.ModeKeys.TRAIN else False
        training_params = params['training']

        output = model(features, params['model'], params['example_description'], training)

        logits = {}
        tasks_dict = params['model']['tasks']
        for task in tasks_dict:
            tasks_dict[task].update({'name': task})
            if task == 'particletype':
                expected_logits_dimension = len(tasks_dict[task]['class_names'])
                classification_logit = fc_head(output, tasks_dict[task], expected_logits_dimension)
                logits['particletype_probabilities'] = tf.nn.softmax(classification_logit)
                logits[task] = tf.cast(tf.argmax(logits['particletype_probabilities'], axis=1),
                                            tf.int32, name="predicted_classes")
                for i, name in enumerate(tasks_dict['particletype']['class_names']):
                    logits[name] = logits['particletype_probabilities'][:, i]
            else:
                expected_logits_dimension = 2 if task in ['direction', 'delta_direction', 'impact'] else 1
                logits[task] = fc_head(output, tasks_dict[task], expected_logits_dimension)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)

        training_params = params['training']
        losses = []
        for task in tasks_dict:
            if task == 'particletype':
                particletype = tf.cast(labels['particletype'], tf.int32,
                                   name="true_classes")
                # Get class weights
                if training_params['apply_class_weights']:
                    class_weights = tf.constant(training_params['class_weights'],
                                                dtype=tf.float32, name="class_weights")
                    weights = tf.gather(class_weights, particletype, name="weights")
                else:
                    weights = 1.0
                num_classes = len(tasks_dict['particletype']['class_names'])
                onehot_labels = tf.one_hot(indices=particletype, depth=num_classes)
                # compute cross-entropy loss
                loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                       logits=classification_logit, weights=weights)

                # add regularization loss
                regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                loss = tf.add_n([loss] + regularization_losses)
                task_loss = tf.math.multiply(loss, tf.cast(tasks_dict[task]['weight'], tf.float32))
            else:
                loss = tf.losses.absolute_difference(
                           labels[task],
                           logits[task],
                           weights=1.0,
                           loss_collection=tf.GraphKeys.LOSSES,
                           reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
                       )

                # add regularization loss
                regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                loss = tf.add_n([loss] + regularization_losses)
                task_loss = tf.math.multiply(loss, tf.cast(tasks_dict[task]['weight'], tf.float32))
            losses.append(task_loss)

        # Combine the losses
        merged_loss = tf.math.add_n(losses)
        # add regularization loss
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        final_loss = tf.add_n([merged_loss] + regularization_losses)

        # Scale the learning rate so batches with fewer triggered
        # telescopes don't have smaller gradients
        # Only apply learning rate scaling for array-level models
        if (training_params['scale_learning_rate'] and params['model']['function'] in ['cnn_rnn_model', 'variable_input_model']):
            trigger_rate = tf.reduce_mean(tf.cast(
                            features['telescope_triggers'], tf.float32),
                            name="trigger_rate")
            trigger_rate = tf.maximum(trigger_rate, 0.1) # Avoid division by 0
            scaling_factor = tf.reciprocal(trigger_rate, name="scaling_factor")
            learning_rate = tf.multiply(scaling_factor,
                                        training_params['base_learning_rate'],
                                        name="learning_rate")
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
            train_op = optimizer.minimize(
                final_loss,
                global_step=tf.train.get_global_step(),
                var_list=var_list)

        # Define the evaluation metrics
        eval_metric_ops = {}
        if config['Evaluation']['tensorflow']:
            for task in tasks_dict:
                if task == 'particletype':
                    eval_metric_ops['accuracy'] = tf.metrics.accuracy(particletype,
                                                                      logits[task])
                    eval_metric_ops['auc'] = tf.metrics.auc(particletype,
                                                            logits['gamma'])

                    # add class-wise accuracies
                    for i, name in enumerate(tasks_dict['particletype']['class_names']):
                        weights = tf.cast(tf.equal(particletype, tf.constant(i)), tf.int32)
                        eval_metric_ops['accuracy_' + name] = tf.metrics.accuracy(
                            particletype, logits[task], weights=weights)
                else:
                    eval_metric_ops['mae_' + task] = tf.metrics.mean_absolute_error(
                           labels[task], logits[task], weights=None)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=final_loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

    # Control tf checkpoints
    tf_config = None
    save_checkpoints_steps = config['Checkpoints'].get('save_checkpoints_steps', None)
    if save_checkpoints_steps:
        tf_config = tf.estimator.RunConfig(save_checkpoints_steps=save_checkpoints_steps)

    estimator = tf.estimator.Estimator(
        model_fn,
        model_dir=model_dir,
        config=tf_config,
        params=params)

    hooks = None
    # Activate Tensorflow debugger if appropriate option set
    if run_tfdbg:
        if not isinstance(hooks, list):
            hooks = []
        hooks.append(tf_debug.LocalCLIDebugHook())

    if mode == 'train':

        # Train and evaluate the model
        logger.info("Training and evaluating...")

        max_steps = int(config['Training']['num_epochs']
                        * num_training_examples / batch_size)

        eval_steps = config['Evaluation'].get('eval_steps', None)
        start_delay_secs = int(config['Evaluation'].get('start_delay_secs', 120))
        throttle_secs = int(config['Evaluation'].get('throttle_secs', 600))

        train_input_fn = lambda: input_fn(reader, training_indices,
                                          shuffle_and_repeat=True,
                                          **config['Input'])
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                            max_steps=max_steps, hooks=hooks)
        eval_input_fn = lambda: input_fn(reader, validation_indices,
                                         **config['Input'])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                          steps=eval_steps,
                                          start_delay_secs=start_delay_secs,
                                          throttle_secs=throttle_secs)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    elif mode == 'predict':

        # Generate predictions and add to output
        logger.info("Predicting...")
        predict_input_fn = lambda: input_fn(reader, indices,
                                            **config['Input'])

        predictions = list(estimator.predict(predict_input_fn))

        output_file = file = config['Prediction'].get('file', "experiment")
        if random_seed:
            file += "_{}".format(random_seed)
        output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), model_dir+"/{}.h5".format(file)))

        write_output(output_file,
                     reader,
                     indices,
                     params['example_description'],
                     predictions,
                     config['Prediction']['prediction_label'])

    # clear the handlers, shutdown the logging and delete the logger
    logger.handlers.clear()
    logging.shutdown()
    del logger
    return

def main():

    parser = argparse.ArgumentParser(
        description=("Train/Predict with a CTLearn model."))
    parser.add_argument(
        '--input', '-i',
        help='input directory (not required when set in the config file)')
    parser.add_argument(
        '--pattern', '-p',
        help='pattern to mask unwanted files from the data input directory',
        default=["*.h5"],
        nargs='+')
    parser.add_argument(
        '--mode', '-m',
        default="train",
        help="Mode to run in (train/predict/train_and_predict/load_only)")
    parser.add_argument(
        'config_file',
        help="path to YAML configuration file with training options")
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help="print debug/logger messages")
    parser.add_argument(
        '--log_to_file', '-l',
        action='store_true',
        help="log to a file in model directory instead of terminal")
    parser.add_argument(
        '--random_seed', '-s',
        default=0,
        type=int,
        help="overwrite the random seed")

    args = parser.parse_args()

    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Overwrite the random seed in the config file
    if args.random_seed != 0:
        if 1000 <= args.random_seed <= 9999:
            config['Data']['seed'] = args.random_seed
            config['Logging']['add_seed'] = True
        else:
            raise ValueError("Random seed: '{}'. "
                             "Must be 4 digit integer!".format(
                             args.random_seed))
    random_seed = config['Data']['seed']

    if args.mode == 'load_only':
        run_model(config, mode=args.mode, debug=args.debug, log_to_file=args.log_to_file)

    if 'train' in args.mode:
        run_model(config, mode='train', debug=args.debug, log_to_file=args.log_to_file)

    if 'predict' in args.mode:
        if args.input:
            abs_file_dir = os.path.abspath(args.input)
            input_data = []
            for pattern in args.pattern:
                files = glob.glob(os.path.join(abs_file_dir, pattern))
                if not files: continue
                for file in files:

                    with open(args.config_file, 'r') as config_file:
                        config = yaml.safe_load(config_file)
                    config['Data']['seed'] = random_seed
                    if args.random_seed != 0:
                        config['Logging']['add_seed'] = True
                    config['Data']['shuffle'] = False

                    config['Prediction']['file'] = file.split("/")[-1].replace("_S_", "_E_").replace("dl1", "dl2").replace(".h5","")
                    config['Prediction']['prediction_label'] = 'data'
                    config['Prediction']['prediction_file_lists'] = {'data': file}
                    run_model(config, mode='predict', debug=args.debug, log_to_file=args.log_to_file)
        else:
            for key in config['Prediction']['prediction_file_lists']:
                with open(args.config_file, 'r') as config_file:
                    config = yaml.safe_load(config_file)
                config['Data']['seed'] = random_seed
                if args.random_seed != 0:
                    config['Logging']['add_seed'] = True
                config['Data']['shuffle'] = False
                config['Prediction']['prediction_label'] = key
                run_model(config, mode='predict', debug=args.debug, log_to_file=args.log_to_file)

if __name__ == "__main__":
    main()
