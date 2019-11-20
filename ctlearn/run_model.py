import argparse
import importlib
import logging
import math
from random import randint
import os
from pprint import pformat
import sys
import time

import numpy as np
import tables
import pkg_resources
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import yaml

from dl1_data_handler.reader_legacy import DL1DataReader
#from dl1_data_handler.reader import DL1DataReader

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
    
def write_output(h5file, reader, indices, learning_tasks, epoch=None, predictions=None, mode='train', format='GammaBoard', seed= None):
    if format == 'GammaBoard':
        # Open the hdf5 file and create the table structure for the hdf5 file.
        if mode == 'train':
            h5 = tables.open_file(h5file, mode="a", title="Final evaluation")
            columns_dict={"mc_particle":tables.Int32Col(pos=0),
                          "reco_particle":tables.Int32Col(pos=1),
                          "reco_gammaness":tables.Float32Col(pos=2),
                          "mc_energy":tables.Float32Col(pos=3),
                          "reco_energy":tables.Float32Col(pos=4),
                          "mc_impact_x":tables.Float32Col(pos=5),
                          "mc_impact_y":tables.Float32Col(pos=6),
                          "reco_impact_x":tables.Float32Col(pos=7),
                          "reco_impact_y":tables.Float32Col(pos=8),
                          "mc_altitude":tables.Float32Col(pos=9),
                          "mc_azimuth":tables.Float32Col(pos=10),
                          "reco_altitude":tables.Float32Col(pos=11),
                          "reco_azimuth":tables.Float32Col(pos=12)}
        elif mode == 'predict':
            h5 = tables.open_file(h5file, mode="a", title="Prediction")
            columns_dict={"event_id":tables.Int32Col(pos=0),
                          "obs_id":tables.Int32Col(pos=1),
                          "tel_id":tables.Int32Col(pos=2),
                          "mc_particle":tables.Int32Col(pos=3),
                          "reco_particle":tables.Int32Col(pos=4),
                          "reco_gammaness":tables.Float32Col(pos=5),
                          "mc_energy":tables.Float32Col(pos=6),
                          "reco_energy":tables.Float32Col(pos=7),
                          "mc_impact_x":tables.Float32Col(pos=8),
                          "mc_impact_y":tables.Float32Col(pos=9),
                          "reco_impact_x":tables.Float32Col(pos=10),
                          "reco_impact_y":tables.Float32Col(pos=11),
                          "mc_altitude":tables.Float32Col(pos=12),
                          "mc_azimuth":tables.Float32Col(pos=13),
                          "reco_altitude":tables.Float32Col(pos=14),
                          "reco_azimuth":tables.Float32Col(pos=15)}
        description = type("description", (tables.IsDescription,), columns_dict)
                    
        # Create the table.
        table_name = "experiment"
        if seed:
            table_name += "_{}".format(seed)
        if "/{}".format(table_name) not in h5:
            table = h5.create_table(eval("h5.root"),table_name,description)
        else:
            eval("h5.root.{}".format(table_name)).remove_rows()

        # Fill the data into the table of the hdf5 file.
        i = 0
        for idx in indices:
            table = eval("h5.root.{}".format(table_name))
            row = table.row
            if mode == 'train':
                row['mc_energy'] = np.log(reader[idx][2])
                row['reco_energy'] = predictions[i][('energy', 'predictions')] if 'energy_regression' in learning_tasks else np.nan
                row['mc_impact_x'] = 0.001*reader[idx][5]
                row['reco_impact_x'] = predictions[i][('impact', 'predictions')][0] if 'impact_regression' in learning_tasks else np.nan
                row['mc_impact_y'] = 0.001*reader[idx][6]
                row['reco_impact_y'] = predictions[i][('impact', 'predictions')][1] if 'impact_regression' in learning_tasks else np.nan
                row['mc_altitude'] = reader[idx][3]
                row['reco_altitude'] = predictions[i][('direction', 'predictions')][0] if 'direction_regression' in learning_tasks else np.nan
                row['mc_azimuth'] = reader[idx][4]
                row['reco_azimuth'] = predictions[i][('direction', 'predictions')][1] if 'direction_regression' in learning_tasks else np.nan
                row['mc_particle'] = reader[idx][1]
                row['reco_particle'] = predictions[i][('particle_type', 'class_ids')][0] if 'gammahadron_classification' in learning_tasks else np.nan
                row['reco_gammaness'] = predictions[i][('particle_type', 'probabilities')][1] if 'gammahadron_classification' in learning_tasks else np.nan
            elif mode == 'predict':
                row['event_id'] = reader[idx][8]
                row['obs_id'] = reader[idx][9]
                row['tel_id'] = reader[idx][1]
                row['mc_energy'] = np.log(reader[idx][3])
                row['reco_energy'] = predictions[i][('energy', 'predictions')] if 'energy_regression' in learning_tasks else np.nan
                row['mc_impact_x'] = 0.001*reader[idx][6]
                row['reco_impact_x'] = predictions[i][('impact', 'predictions')][0] if 'impact_regression' in learning_tasks else np.nan
                row['mc_impact_y'] = 0.001*reader[idx][7]
                row['reco_impact_y'] = predictions[i][('impact', 'predictions')][1] if 'impact_regression' in learning_tasks else np.nan
                row['mc_altitude'] = reader[idx][4]
                row['reco_altitude'] = predictions[i][('direction', 'predictions')][0] if 'direction_regression' in learning_tasks else np.nan
                row['mc_azimuth'] = reader[idx][5]
                row['reco_azimuth'] = predictions[i][('direction', 'predictions')][1] if 'direction_regression' in learning_tasks else np.nan
                row['mc_particle'] = reader[idx][2]
                row['reco_particle'] = predictions[i][('particle_type', 'class_ids')][0] if 'gammahadron_classification' in learning_tasks else np.nan
                row['reco_gammaness'] = predictions[i][('particle_type', 'probabilities')][1] if 'gammahadron_classification' in learning_tasks else np.nan
            row.append()
            table.flush()
            i += 1
        # Close hdf5 file.
        h5.close()
        
    elif format == 'CTLearn_standard':
        # Open the hdf5 evaluation file.
        h5 = tables.open_file(h5file, mode="a", title="Evaluation per epoch")
        if predictions == None:
            # Create the table structure for the hdf5 evaluation file.
            columns_dict={"mc_particle":tables.Int32Col(pos=0),
                          "mc_energy":tables.Float32Col(pos=1),
                          "mc_direction":tables.Float32Col(shape=(2,), pos=2),
                          "mc_impact":tables.Float32Col(shape=(2,), pos=3)}
            description = type("description", (tables.IsDescription,), columns_dict)
                        
            # Create the table for the evaluation of the epoch.
            table_name = "simu_values"
            if "/{}".format(table_name) not in h5:
                table = h5.create_table(eval("h5.root"),table_name,description)
            else:
                eval("h5.root.{}".format(table_name)).remove_rows()

            # Fill the data of the final evaluation into the table of the hdf5 file.
            i = 0
            for idx in indices:
                table = eval("h5.root.{}".format(table_name))
                row = table.row
                row['mc_particle'] = reader[idx][1]
                row['mc_energy'] = np.log(reader[idx][2])
                row['mc_direction'] = [reader[idx][3], reader[idx][4]]
                row['mc_impact'] = [0.001*reader[idx][5], 0.001*reader[idx][6]]
                row.append()
                table.flush()
                i += 1
        else:
            # Create the table structure for the hdf5 evaluation file.
            columns_dict={"reco_particle":tables.Int32Col(pos=0),
                          "reco_gammaness":tables.Float32Col(pos=1),
                          "reco_energy":tables.Float32Col(pos=2),
                          "reco_direction":tables.Float32Col(shape=(2,), pos=3),
                          "reco_impact":tables.Float32Col(shape=(2,), pos=4)}
            description = type("description", (tables.IsDescription,), columns_dict)
                            
            # Create the table for the evaluation of the epoch.
            table_name = "reco_values_epoch_{}".format(epoch)
            if "/{}".format(table_name) not in h5:
                table = h5.create_table(eval("h5.root"),table_name,description)
            else:
                eval("h5.root.{}".format(table_name)).remove_rows()

            # Fill the data of the final evaluation into the table of the hdf5 file.
            i = 0
            for idx in indices:
                table = eval("h5.root.{}".format(table_name))
                row = table.row
                row['reco_particle'] = predictions[i][('particle_type', 'class_ids')][0] if 'gammahadron_classification' in learning_tasks else np.nan
                row['reco_gammaness'] = predictions[i][('particle_type', 'probabilities')][1] if 'gammahadron_classification' in learning_tasks else np.nan
                row['reco_energy'] = predictions[i][('energy', 'predictions')] if 'energy_regression' in learning_tasks else np.nan
                row['reco_direction'] = predictions[i][('direction', 'predictions')] if 'direction_regression' in learning_tasks else [np.nan, np.nan]
                row['reco_impact'] = predictions[i][('impact', 'predictions')] if 'impact_regression' in learning_tasks else [np.nan, np.nan]
                row.append()
                table.flush()
                i += 1
        # Close hdf5 eval file.
        h5.close()

def run_model(config, mode="train", debug=False, log_to_file=False, multiple_runs=1):

    # Load options relating to logging and checkpointing
    root_model_dir = model_dir = config['Logging']['model_directory']
    # Create model directory if it doesn't exist already
    if not os.path.exists(root_model_dir):
        if mode == 'predict':
            raise ValueError("Invalid model directory '{}'. "
            "Must be a path to an existing directory in the predict mode.".format(config['Logging']['model_directory']))
        os.makedirs(root_model_dir)

    random_seed = None
    if multiple_runs != 1:
        random_seed = config['Data']['seed']
        if mode=='train':
            model_dir += "/experiment_{}".format(random_seed)
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

    # Parse file list or prediction file list
    if mode == 'train':
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
        if isinstance(config['Prediction']['prediction_file_list'], str):
            data_files = []
            with open(config['Prediction']['prediction_file_list']) as f:
                for line in f:
                    line = line.strip()
                    if line and line[0] != "#":
                        data_files.append(line)
            config['Data']['file_list'] = data_files
        if not isinstance(config['Data']['file_list'], list):
            raise ValueError("Invalid prediction file list '{}'. "
                             "Must be list or path to file".format(config['Prediction']['prediction_file_list']))
                             
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
    if 'interpolation_image_shape' in config['Data'].get('mapping_settings',{}):
        config['Data']['mapping_settings']['interpolation_image_shape'] = {
            k: tuple(l) for k, l in config['Data']['mapping_settings']['interpolation_image_shape'].items()}

    config['Data']['event_info'] = ['particle_id', 'mc_energy', 'alt', 'az', 'core_x', 'core_y']
    #config['Data']['event_info'] = ['particle_id', 'mc_energy', 'alt', 'az', 'core_x', 'core_y', 'x_max']
    
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
            #config['Data']['event_info'].extend(['event_id', 'obs_id'])
            #if config['Data']['mode'] == 'mono':
            #    if 'array_info' not in config['Data']:
            #        config['Data']['array_info'] = []
            #    config['Data']['array_info'].append('id')

    # Load learning tasks according to the selected model
    if params['model']['model']['function'] == 'vanilla_model':
        learning_tasks = params['model']['learning_tasks']
    elif params['model']['model']['function'] == 'gammaPhysNet_model':
        learning_tasks = ['gammahadron_classification', 'energy_regression', 'direction_regression', 'impact_regression']
    elif params['model']['model']['function'] == 'gammaPhysNet2_model':
        learning_tasks = ['gammahadron_classification', 'energy_regression', 'direction_regression', 'impact_regression', 'showermaximum_regression']
    elif params['model']['model']['function'] == 'gammaPhysNetS_model':
        learning_tasks = ['gammahadron_classification', 'energy_regression', 'direction_regression', 'impact_regression']
    elif params['model']['model']['function'] == 'gammaPhysNet2S_model':
        learning_tasks = ['gammahadron_classification', 'energy_regression', 'direction_regression', 'impact_regression', 'showermaximum_regression']
    else:
        raise ValueError("Invalid model selection '{}'. Valid options: 'vanilla_classification',"
                "'vanilla', 'gammaPhysNet', 'gammaPhysNet2', 'gammaPhysNetS', 'gammaPhysNet2S'".format(params['model']['model']['module']))

    learning_task_labels = {}
    if 'gammahadron_classification' in learning_tasks:
        learning_task_labels['class_label'] = params['model']['label_names']['class_label']
    if 'energy_regression' in learning_tasks:
        learning_task_labels['mc_energy'] = 'Simulated (MC) Primary Particle Energy'
    if 'direction_regression' in learning_tasks:
        learning_task_labels['alt'] = 'Zenith Angle'
        learning_task_labels['az'] = 'Azimuth Angle'
    if 'impact_regression' in learning_tasks:
        learning_task_labels['core_x'] = 'Shower core x-position'
        learning_task_labels['core_y'] = 'Shower core y-position'
    if 'showermaximum_regression' in learning_tasks:
        learning_task_labels['x_max'] = 'Location of shower maximum'
    
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
    config['Input']['label_names'].update(learning_task_labels)
    
    # Load either training or prediction options
    # and log information about the data set
    indices = list(range(len(reader)))
    labels = config['Model'].get('label_names', {})

    batch_size = config['Input'].get('batch_size', 1)
    logger.info("Batch size: {}".format(batch_size))
    
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
        logger.info("Number of training steps per epoch: {}".format(
            int(num_training_examples / batch_size)))
        logger.info("Number of training steps between validations: {}".format(
            config['Training']['num_training_steps_per_validation']))
                    
        # Write the evaluation configuration in the params dict
        params['evaluation'] = config['Evaluation']
        # Write the true labels to the hdf5 evaluation file
        if params['evaluation']['custom_ctlearn']['evaluation_per_epoch']:
            logger.info("Write the true labels to the hdf5 evaluation file...")
            # Open the the h5 to dump the evaluation information in the selected format
            if params['evaluation']['custom_ctlearn']['evaluation_file_name']:
                eval_file = os.path.abspath(os.path.join(os.path.dirname(__file__), model_dir+"/{}.h5".format(params['evaluation']['custom_ctlearn']['evaluation_file_name'])))
            else:
                eval_file = os.path.abspath(os.path.join(os.path.dirname(__file__), model_dir+"/ctlearn_evaluation.h5"))
            write_output(h5file=eval_file, reader=reader, indices=validation_indices, learning_tasks=learning_tasks, format='CTLearn_standard')

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

        return features, labels

    # Define model function with model, mode (train/predict),
    # metrics, optimizer, learning rate, etc.
    # to pass into TF Estimator
    def model_fn(features, labels, mode, params):
        return model(features, labels, mode, params)
    
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
            epoch = num_validations-num_validations_remaining+1
            estimator.train(
                lambda: input_fn(reader, training_indices, mode='train', **config['Input']),
                steps=steps, hooks=hooks)
            if params['evaluation']['default_tensorflow']:
                logger.info("Evaluate with the default TensorFlow evaluation...")
                estimator.evaluate(
                    lambda: input_fn(reader, validation_indices, mode='eval', **config['Input']),
                    hooks=hooks, name='validation')

            if params['evaluation']['custom_ctlearn']['final_evaluation'] and num_validations_remaining == 1:
                logger.info("Evaluate with the custom CTLearn evaluation...")
                evaluations = estimator.predict(
                    lambda: input_fn(reader, validation_indices, mode='predict', **config['Input']),
                    hooks=hooks)

                evaluation = list(evaluations)
                
                # Open the the h5 to dump the final evaluation information in the selected format
                if params['evaluation']['custom_ctlearn']['final_evaluation_file_name']:
                    final_eval_file = os.path.abspath(os.path.join(os.path.dirname(__file__), model_dir+"/{}.h5".format(params['evaluation']['custom_ctlearn']['final_evaluation_file_name'])))
                else:
                    final_eval_file = os.path.abspath(os.path.join(os.path.dirname(__file__), model_dir+"/experiment.h5"))
                write_output(h5file=final_eval_file, reader=reader, indices=validation_indices, learning_tasks=learning_tasks, predictions=evaluation)
            
            if params['evaluation']['custom_ctlearn']['evaluation_per_epoch']:
                logger.info("Evaluate with the custom CTLearn evaluation...")
                evaluations = estimator.predict(
                    lambda: input_fn(reader, validation_indices, mode='predict', **config['Input']),
                    hooks=hooks)

                evaluation = list(evaluations)
                write_output(h5file=eval_file, reader=reader, indices=validation_indices, learning_tasks=learning_tasks, epoch=epoch, predictions=evaluation, format='CTLearn_standard')

            if not train_forever:
                num_validations_remaining -= 1

    elif mode == 'predict':

        # Generate predictions and add to output
        logger.info("Predicting...")
        
        # Open the the h5 to dump the final evaluation information in the selected format
        if params['prediction']['prediction_file_name']:
            predict_file = os.path.abspath(os.path.join(os.path.dirname(__file__), root_model_dir+"/{}.h5".format(params['prediction']['prediction_file_name'])))
        else:
            predict_file = os.path.abspath(os.path.join(os.path.dirname(__file__), root_model_dir+"/ctlearn_prediction.h5"))
        predictions = estimator.predict(
            lambda: input_fn(reader, indices, mode='predict', **config['Input']),
            hooks=hooks)
        prediction = list(predictions)
        write_output(h5file=predict_file, reader=reader, indices=indices, learning_tasks=learning_tasks, predictions=prediction, mode='predict', format='GammaBoard', seed=random_seed)
        
    # clear the handlers, shutdown the logging and delete the logger
    logger.handlers.clear()
    logging.shutdown()
    del logger
    return
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("Train/Predict with a CTLearn model."))
    parser.add_argument(
        '--mode',
        default="train",
        help="Mode to run in (train/predict/trainandpredict)")
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
    parser.add_argument(
        '--multiple_runs',
        default=1,
        type=int,
        help="")

    args = parser.parse_args()

    random_seeds = []
    for run in np.arange(args.multiple_runs):
        with open(args.config_file, 'r') as config_file:
            config = yaml.safe_load(config_file)
        if args.multiple_runs != 1:
            # Create and overwrite the random seed in the config file
            while True:
                random_seed = randint(1000,9999)
                if random_seed not in random_seeds:
                    random_seeds.append(random_seed)
                    break
            
            config['Data']['seed'] = random_seed
            print("CTLearn run {} with random seed '{}':".format(run+1,config['Data']['seed']))
        config['Data']['shuffle'] = False if args.mode == 'predict' else True

        if args.mode in ['train', 'predict']:
            run_model(config, mode=args.mode, debug=args.debug, log_to_file=args.log_to_file, multiple_runs=args.multiple_runs)
        else:
            run_model(config, mode='train', debug=args.debug, log_to_file=args.log_to_file, multiple_runs=args.multiple_runs)
            with open(args.config_file, 'r') as config_file:
                config = yaml.safe_load(config_file)
            if args.multiple_runs != 1:
                config['Data']['seed'] = random_seed
            config['Data']['shuffle'] = False
            run_model(config, mode='predict', debug=args.debug, log_to_file=args.log_to_file, multiple_runs=args.multiple_runs)
