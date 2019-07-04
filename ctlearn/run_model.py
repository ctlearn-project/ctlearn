import argparse
import importlib
import logging
import math
import os
from pprint import pformat
import sys
import time

import numpy as np
import pkg_resources
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import yaml

from dl1_data_handler.reader_legacy import DL1DataReader

# Disable Tensorflow info and warning messages (not error messages)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.logging.set_verbosity(tf.logging.WARN)

def setup_logging(config, log_dir, debug, log_to_file):

    # Log configuration to a text file in the log dir
    time_str = time.strftime('%Y%m%d_%H%M%S')
    config_filename = os.path.join(log_dir, time_str + '_config.yml')
    with open(config_filename, 'w') as outfile:
        ctlearn_version = pkg_resources.get_distribution("ctlearn").version
        outfile.write('# Training performed with '
                      'CTLearn version {}.\n'.format(ctlearn_version))
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

def compute_class_weights(labels, num_class_examples):
    logger = logging.getLogger()
    class_weights = []
    total_num = sum(num_class_examples.values())
    logger.info("Computing class weights...")
    for idx, class_name in enumerate(labels['class_label']):
        try:
            num = num_class_examples[(idx,)]
            class_inverse_frac = total_num / num
            class_weights.append(class_inverse_frac)
        except KeyError:
            logger.warning("Class '{}' has no examples, unable to "
                           "calculate class weights".format(class_name))
            class_weights = [1.0 for l in labels['class_label']]
            break
    logger.info("Class weights: {}".format(class_weights))
    return class_weights

def load_from_module(name, module, path=None, args=None):
    if path is not None and path not in sys.path:
        sys.path.append(path)
    mod = importlib.import_module(module)
    fn = getattr(mod, name)
    params = args if args is not None else {}
    return fn, params

def log_examples(reader, indices, labels, subset_name):
    logger = logging.getLogger()
    logger.info("Examples for " + subset_name + ':')
    logger.info("  Total number: {}".format(len(indices)))
    label_list = list(labels)
    num_class_examples = reader.num_examples(group_by=label_list,
                                             example_indices=indices)
    logger.info("  Breakdown by " + ', '.join(label_list) + ':')
    for cls, num in num_class_examples.items():
        names = []
        for label, idx in zip(label_list, cls):
            # Value if regression label, else class name if class label
            name = str(idx) if labels[label] is None else labels[label][idx]
            names.append(name)
        logger.info("    " + ', '.join(names) + ": {}".format(num))
    logger.info('')
    return num_class_examples

def run_model(config, mode="train", debug=False, log_to_file=False):

    # Load options relating to logging and checkpointing
    model_dir = config['Logging']['model_directory']
    # Create model directory if it doesn't exist already
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
    model_type = config['Data'].get('Loading', {}).get('example_type', 'array')

    params['model'] = {**config['Model'], **config.get('Model Parameters', {})}

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

    # Parse list of event selection filters
    event_selection = {}
    for s in config['Data'].get('event_selection', {}):
        s = {'module': 'dl1_data_handler.utils', **s}
        filter_fn, filter_params = load_from_module(**s)
        event_selection[filter_fn] = filter_params
    config['Data']['event_selection'] = event_selection

    # Parse list of image selection filters
    image_selection = {}
    for s in config['Data'].get('image_selection', {}):
        s = {'module': 'dl1_data_handler.utils', **s}
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
    if 'interpolation_image_shape' in config['Data'].get('mapping_settings',
                                                         {}):
        config['Data']['mapping_settings']['interpolation_image_shape'] = {
            k: tuple(l) for k, l in config['Data']['mapping_settings']['interpolation_image_shape'].items()}

    # Possibly add additional info to load if predicting to write later
    if mode == 'predict':

        if 'Prediction' not in config:
            config['Prediction'] = {}
        params['prediction'] = config['Prediction']

        if config['Prediction'].get('save_identifiers', False):
            if 'event_info' not in config['Data']:
                config['Data']['event_info'] = []
            config['Data']['event_info'].extend(['event_number', 'run_number'])
            if config['Data']['mode'] == 'mono':
                if 'array_info' not in config['Data']:
                    config['Data']['array_info'] = []
                config['Data']['array_info'].append('tel_id')

    # Create data reader
    logger.info("Loading data:")
    logger.info("For a large dataset, this may take a while...")
    reader = DL1DataReader(**config['Data'])
    params['example_description'] = reader.example_description

    # Define format for TensorFlow dataset
    if 'Input' not in config:
        config['Input'] = {}
    config['Input']['output_names'] = [d['name'] for d
                                       in reader.example_description]
    # TensorFlow does not support conversion for NumPy unsigned dtypes
    # other than int8. Work around this by doing a manual conversion.
    dtypes = [d['dtype'] for d in reader.example_description]
    for i, dtype in enumerate(dtypes):
        for utype, stype in [(np.uint16, np.int32), (np.uint32, np.int64)]:
            if dtype == utype:
                dtypes[i] = stype
    config['Input']['output_dtypes'] = tuple(tf.as_dtype(d) for d in dtypes)
    config['Input']['output_shapes'] = tuple(tf.TensorShape(d['shape']) for d
                                             in reader.example_description)
    config['Input']['label_names'] = config['Model'].get('label_names', {})

    # Load either training or prediction options
    # and log information about the data set
    indices = list(range(len(reader)))
    labels = config['Model'].get('label_names', {})

    batch_size = config['Input'].get('batch_size', 1)
    logger.info("Batch size: {}".format(batch_size))

    if mode in ['train', 'load_only']:

        params['training'] = config['Training']
        params['training']['model_type'] = model_type

        validation_split = config['Training']['validation_split']
        if not 0.0 < validation_split < 1.0:
            raise ValueError("Invalid validation split: {}. "
                             "Must be between 0.0 and 1.0".format(
                                 validation_split))
        num_training_examples = math.floor((1 - validation_split) * len(reader))
        training_indices = indices[:num_training_examples]
        validation_indices = indices[num_training_examples:]
        logger.info("Number of training steps per epoch: {}".format(
            int(num_training_examples / batch_size)))
        logger.info("Number of training steps between validations: {}".format(
            config['Training']['num_training_steps_per_validation']))

    if mode == 'load_only':

        log_examples(reader, indices, labels, 'total dataset')
        log_examples(reader, training_indices, labels, 'training')
        log_examples(reader, validation_indices, labels, 'validation')
        # If only loading data, can end now that dataset logging is complete
        return

    if mode == 'train' and config['Training']['apply_class_weights']:
        num_class_examples = log_examples(reader, training_indices,
                                          labels, 'training')
        class_weights = compute_class_weights(labels, num_class_examples)
        params['training']['class_weights'] = class_weights

    # Load options for TensorFlow
    run_tfdbg = config.get('TensorFlow', {}).get('run_TFDBG', False)

    # Define input function for TF Estimator
    def input_fn(reader, indices, output_names, output_dtypes, output_shapes,
                 label_names, seed=None, batch_size=1,
                 shuffle_buffer_size=None, prefetch_buffer_size=1,
                 add_labels_to_features=False):

        def generator(indices):
            for idx in indices:
                yield tuple(reader[idx])

        dataset = tf.data.Dataset.from_generator(generator, output_dtypes,
                                                 output_shapes=output_shapes,
                                                 args=(indices,))
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

        if add_labels_to_features:  # for predict mode
            features['labels'] = labels

        return features, labels

    # Define model function with model, mode (train/predict),
    # metrics, optimizer, learning rate, etc.
    # to pass into TF Estimator
    def model_fn(features, labels, mode, params):

        training = True if mode == tf.estimator.ModeKeys.TRAIN else False

        logits = model(features, params['model'],
                       params['example_description'], training)

        # Collect predictions
        predictions = {}
        classifier_values = tf.nn.softmax(logits)
        predicted_classes = tf.cast(tf.argmax(classifier_values, axis=1),
                                    tf.int32, name="predicted_classes")
        predictions['predicted_class'] = predicted_classes
        for i, name in enumerate(params['model']['label_names']['class_label']):
            predictions[name] = classifier_values[:, i]

        # In predict mode, finish up here
        if mode == tf.estimator.ModeKeys.PREDICT:

            # Append the event identifiers to the predictions
            if params['prediction'].get('save_identifiers', False):
                for d in params['example_description']:
                    if d['name'] in ['event_number', 'run_number']:
                        predictions[d['name']] = features[d['name']]
                    if d['name'] == 'tel_id':
                        predictions['tel_id'] = tf.reshape(features[d['name']],
                                                           [-1])
            # Append the labels to the predictions
            if params['prediction'].get('save_labels', False):
                for key, value in features['labels'].items():
                    predictions[key] = value

            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        training_params = params['training']

        # Compute class-weighted softmax-cross-entropy
        true_classes = tf.cast(labels['class_label'], tf.int32,
                               name="true_classes")

        # Get class weights
        if training_params['apply_class_weights']:
            class_weights = tf.constant(training_params['class_weights'],
                                        dtype=tf.float32, name="class_weights")
            weights = tf.gather(class_weights, true_classes, name="weights")
        else:
            weights = 1.0

        num_classes = len(params['model']['label_names']['class_label'])
        onehot_labels = tf.one_hot(indices=true_classes, depth=num_classes)

        # compute cross-entropy loss
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                               logits=logits, weights=weights)

        # add regularization loss
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([loss] + regularization_losses, name="loss")

        # Compute accuracy
        training_accuracy = tf.reduce_mean(tf.cast(tf.equal(true_classes,
                                                            predicted_classes),
                                                   tf.float32),
                                           name="training_accuracy")
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
                loss,
                global_step=tf.train.get_global_step(),
                var_list=var_list)

        # Define the evaluation metrics
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(true_classes,
                                            predicted_classes),
            'auc': tf.metrics.auc(true_classes, predictions['gamma'])
            }

        # add class-wise accuracies
        for i, name in enumerate(params['model']['label_names']['class_label']):
            weights = tf.cast(tf.equal(true_classes, tf.constant(i)), tf.int32)
            eval_metric_ops['accuracy_' + name] = tf.metrics.accuracy(
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
        logger.info("Training and evaluating...")
        num_validations = config['Training']['num_validations']
        steps = config['Training']['num_training_steps_per_validation']
        train_forever = False if num_validations != 0 else True
        num_validations_remaining = num_validations
        while train_forever or num_validations_remaining:
            estimator.train(
                lambda: input_fn(reader, training_indices, **config['Input']),
                steps=steps, hooks=hooks)
            estimator.evaluate(
                lambda: input_fn(reader, validation_indices, **config['Input']),
                hooks=hooks, name='validation')
            if not train_forever:
                num_validations_remaining -= 1

    elif mode == 'predict':

        # Generate predictions and add to output
        logger.info("Predicting...")

        if config['Prediction'].get('save_labels', False):
            config['Input']['add_labels_to_features'] = True

        predictions = estimator.predict(
            lambda: input_fn(reader, indices, **config['Input']),
            hooks=hooks)

        # Write predictions and other info given a dictionary of input, with
        # the key:value pairs of header name: list of the values for each event
        def write_predictions(file_handle, predictions):

            def write(prediction):
                row = ",".join('{}'.format(v) for v in prediction.values())
                row += '\n'
                file_handle.write(row)

            prediction = next(predictions)
            header = ",".join([key for key in prediction]) + '\n'
            file_handle.write(header)
            write(prediction)
            for prediction in predictions:
                write(prediction)

        # Write predictions to a csv file
        if config['Prediction'].get('export_as_file', False):
            prediction_path = config['Prediction']['prediction_file_path']
            with open(prediction_path, 'w') as predict_file:
                write_predictions(predict_file, predictions)
        else:
            write_predictions(sys.stdout, predictions)

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
