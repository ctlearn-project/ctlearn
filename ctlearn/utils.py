import importlib
import logging
import os
import pkg_resources
import sys
import time

import numpy as np
import pandas as pd
import yaml

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
    logger.info("Computing class weights...")
    total_num = sum(num_class_examples.values())
    class_weights = []
    for idx, class_name in enumerate(labels):
        try:
            num = num_class_examples[(idx,)]
            class_inverse_frac = total_num / num
            class_weights.append(class_inverse_frac)
        except KeyError:
            logger.warning("Class '{}' has no examples, unable to "
                           "calculate class weights".format(class_name))
            class_weights = [1.0 for l in labels]
            break
    logger.info("Class labels: {}".format(labels))
    logger.info("Class weights: {}".format(class_weights))
    return class_weights

def load_from_module(name, module, path=None, args=None):
    if path is not None and path not in sys.path:
        sys.path.append(path)
    mod = importlib.import_module(module)
    fn = getattr(mod, name)
    params = args if args is not None else {}
    return fn, params

def log_examples(reader, indices, tasks, subset_name, group_by=None):
    """ Log the number of examples in each class or combination

    Specify the names of the labels to group by as a sequence with group_by.
    If group_by is None, group by all labels (default).
    """
    logger = logging.getLogger()
    logger.info("Examples for " + subset_name + ':')
    logger.info("  Total number: {}".format(len(indices)))
    labels = list(tasks)
    if group_by is None:
        group_by = labels
    num_class_examples = reader.num_examples(group_by=group_by,
                                             example_indices=indices)
    logger.info("  Breakdown by " + ', '.join(labels) + ':')
    for cls, num in num_class_examples.items():
        names = []
        for label, idx in zip(labels, cls):
            # Value if regression label, else class name if class label
            if tasks[label].get('class_names') is not None:
                name = tasks[label]['class_names'][idx]
            else:
                name = str(idx)
            names.append(name)
        logger.info("    " + ', '.join(names) + ": {}".format(num))
    logger.info('')
    return num_class_examples
