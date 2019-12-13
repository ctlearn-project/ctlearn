import importlib
import logging
import numpy as np
import os
import pkg_resources
import sys
import tables
import time
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
    class_weights = []
    total_num = sum(num_class_examples.values())
    logger.info("Computing class weights...")
    for idx, class_name in enumerate(labels['particletype']):
        try:
            num = num_class_examples[(idx,)]
            class_inverse_frac = total_num / num
            class_weights.append(class_inverse_frac)
        except KeyError:
            logger.warning("Class '{}' has no examples, unable to "
                           "calculate class weights".format(class_name))
            class_weights = [1.0 for l in labels['particletype']]
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
    
def write_output(h5file, reader, indices, example_description, epoch=None, predictions=None, mode='train', seed=None):

    print(example_description)
    # Open the hdf5 file and create the table structure for the hdf5 file.
    title = "Final evaluation" if mode == 'train' else "Prediction"
        
    h5 = tables.open_file(h5file, mode="a", title=title)
    columns_dict = {}

    for des in example_description:
        if des['name'] == 'particletype':
            columns_dict['mc_particle'] = tables.Int32Col()
            columns_dict['reco_particle'] = tables.Int32Col()
            columns_dict['reco_gammaness'] = tables.Float32Col()
        elif des['name'] == 'mc_energy':
            columns_dict['mc_energy'] = tables.Float32Col()
            columns_dict['reco_energy'] = tables.Float32Col()
        elif des['name'] == 'direction':
            columns_dict['mc_altitude'] = tables.Float32Col()
            columns_dict['reco_altitude'] = tables.Float32Col()
            columns_dict['mc_azimuth'] = tables.Float32Col()
            columns_dict['reco_azimuth'] = tables.Float32Col()
        elif des['name'] == 'impact':
            columns_dict['mc_impact_x'] = tables.Float32Col()
            columns_dict['reco_impact_x'] = tables.Float32Col()
            columns_dict['mc_impact_y'] = tables.Float32Col()
            columns_dict['reco_impact_y'] = tables.Float32Col()
        elif des['name'] == 'event_id':
            columns_dict['event_id'] = tables.Int32Col()
        elif des['name'] == 'obs_id':
            columns_dict['obs_id'] = tables.Int32Col()
        elif des['name'] == 'tel_id':
            columns_dict['tel_id'] = tables.Int32Col()

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
        
        for val, des in zip(reader[idx], example_description):
            if des['name'] == 'particletype':
                row['mc_particle'] = val
                row['reco_particle'] = predictions[i][('particletype', 'class_ids')][0]
                row['reco_gammaness'] = predictions[i][('particletype', 'probabilities')][1]
            elif des['name'] == 'mc_energy':
                row['mc_energy'] = val
                row['reco_energy'] = predictions[i][('mc_energy', 'predictions')]
            elif des['name'] == 'direction':
                row['mc_altitude'] = val[0]
                row['reco_altitude'] = predictions[i][('direction', 'predictions')][0]
                row['mc_azimuth'] = val[1]
                row['reco_azimuth'] = predictions[i][('direction', 'predictions')][1]
            elif des['name'] == 'impact':
                row['mc_impact_x'] = val[0]
                row['reco_impact_x'] = predictions[i][('impact', 'predictions')][0]
                row['mc_impact_y'] = val[1]
                row['reco_impact_y'] = predictions[i][('impact', 'predictions')][1]
            elif des['name'] == 'event_id':
                row['event_id'] = val
            elif des['name'] == 'obs_id':
                row['obs_id'] = val
            elif des['name'] == 'tel_id':
                row['tel_id'] = val
        row.append()
        table.flush()
        i += 1
    # Close hdf5 file.
    h5.close()
