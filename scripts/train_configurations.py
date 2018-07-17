"""
Conduct an automated hyperparameter search using the specified configuration
options.

train_configurations.py <constant_config_path> <changing_configs_path> <run_combinations_path> <model_dir> <combinations_multiplier>

<constant_config_path>: path to YAML configuration file containing all options meant
to be the same for every run. Omit all options which are meant to be changed between runs
(those should only be present in the changing_configs YAML file.

<changing_configs_path>: path to YAML file specifying options to change on each run
The value of each setting (key) in the config dict (which can be nested) should have one of the following forms:
['discrete', [<value1>, <value2>,..., <valueN>]]
['grid_range', [<lower_bound>, <upper_bound>, <spacing>, <num>]]
['random_range', [<lower_bound>, <upper_bound>, <spacing>]]
where <spacing> is one of "linear" or "log".

<run_combinations_path>: path to file in which to save the combinations of 
options corresponding to each run number

<model_dir>: model directory path for storing checkpoints and log files. Each
run will be put in a subdirectory 'run0/', 'run1/', etc. Any
Logging/ModelDirectory specified in specified in <constant_config_path> or
<changing_configs_path> will be overwritten by <model_dir>.

<combinations_multiplier>: number of times to run over each combination. Useful
when some or all parameters have a random iteration type. If all changing
parameters are chosen randomly, <combinations_multiplier> will the total number
of runs.
"""

import argparse
import itertools
import os
import sys

import numpy as np

import yaml

from ctlearn.run_model import run_model

# Get command line arguments
parser = argparse.ArgumentParser(
    description=("Train a ctalearn model over different hyperparameter choices."))
parser.add_argument(
        'constant_config_path',
        help="path to configuration file with constant options")
parser.add_argument(
        'changing_configs_path',
        help="path to file listing configuration options to change")
parser.add_argument(
        'run_combinations_path',
        help="path to file in which to save combinations for each run")
parser.add_argument(
        'model_dir',
        help="model directory path for checkpoints and logging"
        )
parser.add_argument(
        'combinations_multiplier',
        type=int,
        help="number of times to run over each discrete combination")
parser.add_argument(
        '--debug',
        action='store_true',
        help="print debug/logger messages")
parser.add_argument(
        '--log_to_file',
        action='store_true',
        help="log to a file in model directory instead of terminal")

args = parser.parse_args()

def yield_config(constant_config_path, changing_configs, model_dir,
        combinations_multiplier):
    # List tuples of the config coordinates and value for each value of each
    # discrete config
    discrete_configs = [[(c['keys'], v) for v in c['values']]
            for c in changing_configs if not c['random']]
    # Get all the combinations
    discrete_combinations = (list(itertools.product(*discrete_configs)) *
            combinations_multiplier)
    combinations = [d + tuple([(c['keys'], c['value_fn']()) for
        c in changing_configs if c['random']]) for d in discrete_combinations]
    
    for run_num, combination in enumerate(combinations):
        with open(args.constant_config_path, 'r') as constant_config_file:
            config = yaml.load(constant_config_file)
        combination_dict = {}
        # Add the changing config options to the constant config
        for (keys, value) in combination:
            section = config
            combination_dict_section = combination_dict
            for i in range(len(keys) - 1):
                if keys[i] not in section:
                    section[keys[i]] = {}
                section = section[keys[i]]

                if keys[i] not in combination_dict_section:
                    combination_dict_section[keys[i]] = {}
                combination_dict_section = combination_dict_section[keys[i]]

            section[keys[-1]] = value
            combination_dict_section[keys[-1]] = value
        # Set the model directory to that corresponding to this run
        run_model_directory = os.path.join(model_dir, 'run'+str(run_num))
        config['Logging']['model_directory'] = run_model_directory
        yield config, combination_dict

def get_changing_configs(section, keys, changing_configs):
    for k, v in section.items(): 
        if isinstance(v, dict):
            keys.append(k)
            get_changing_configs(v, keys, changing_configs)
        else:
            config = {}
            config['keys'] = keys + [k]
            iteration_type, settings = v 
            if iteration_type == 'discrete':  
                config['random'] = False
                config['values'] = settings
            elif iteration_type == 'grid_range':
                config['random'] = False
                lower_bound, upper_bound, spacing, num = settings
                lower_bound = float(lower_bound)
                upper_bound = float(upper_bound)
                num = int(num)
                if spacing == 'linear':
                    config['values'] = np.linspace(lower_bound, upper_bound, num)
                elif spacing == 'log':
                    config['values'] = np.logspace(np.log10(lower_bound),
                            np.log10(upper_bound), num)
            elif iteration_type == 'random_range':
                config['random'] = True
                lower_bound, upper_bound, spacing = settings
                lower_bound = float(lower_bound)
                upper_bound = float(upper_bound)
                if spacing == 'linear':
                    config['value_fn'] = lambda: np.random.uniform(lower_bound,
                            upper_bound)
                elif spacing == 'log':
                    config['value_fn'] = lambda: np.power(10.0,
                            np.random.uniform(
                                np.log10(lower_bound), np.log10(upper_bound)))

            changing_configs.append(config)

# Load constant config
with open(args.constant_config_path, 'r') as constant_config_file:
    constant_config = yaml.load(constant_config_file)

# Load a file specifying config options to run over
changing_configs = []
keys = []
with open(args.changing_configs_path, 'r') as changing_config_file:
    changing_options_config = yaml.load(changing_config_file)
    get_changing_configs(changing_options_config, keys, changing_configs)
    print(changing_configs)

# write constant part of config
with open(args.run_combinations_path, 'a') as runs_file:
    runs_file.write("Constant settings \n")        
    yaml.dump(constant_config, runs_file, default_flow_style=False)
    runs_file.write("\n")

for run_num, (config, combination_dict) in enumerate(yield_config(
    args.constant_config_path, changing_configs, args.model_dir,
    args.combinations_multiplier)):
    print("Training run {}...".format(run_num))
    # Save the hyperparameter combination for each run to a file
    with open(args.run_combinations_path, 'a') as runs_file:
        runs_file.write("Run {}\n".format(run_num))        
        yaml.dump(combination_dict, runs_file, default_flow_style=False)
        runs_file.write("\n")
    run_model(config, mode='train', debug=args.debug,
            log_to_file=args.log_to_file)
