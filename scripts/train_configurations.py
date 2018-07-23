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
import copy
import os
import sys

import numpy as np
import yaml

from ctlearn.run_model import run_model

# Get command line arguments
parser = argparse.ArgumentParser(
    description=("Train a ctalearn model over different hyperparameter choices."))
parser.add_argument(
        'config_path',
        help="path to configuration file containing base options and Multiple Configurations sections")
parser.add_argument(
        '--mode',
        default="train",
        help="Mode to run in (train/predict)")
parser.add_argument(
        '--debug',
        action='store_true',
        help="print debug/logger messages")
parser.add_argument(
        '--log_to_file',
        action='store_true',
        help="log to a file in model directory instead of terminal")

args = parser.parse_args()

def parse_range_values(range_values, num_grouped_range_values):

    def log_space(lower_bound, upper_bound, num_values):
        return np.logspace(np.log10(lower_bound), np.log10(upper_bound),
                num_values)

    def random_log_uniform(lower_bound, upper_bound, num_values):
        return np.power(10.0, np.random.uniform(np.log10(lower_bound),
            np.log10(upper_bound), num_values))

    settings_to_value_fn = {
            ('linear', 'grid'): np.linspace,
            ('log', 'grid'): log_space,
            ('linear', 'random'): np.random.uniform,
            ('log', 'random'): random_log_uniform
            }

    value_fn = settings_to_value_fn[(range_values['spacing'],
        range_values['selection'])]
    # By default, group all (and only) the random range configurations,
    # overriding this if grouped is set by the user
    grouped = range_values.get('grouped', range_values['selection']=='random')
    if grouped:
        values = value_fn(
                range_values['lower_bound'],
                range_values['upper_bound'],
                num_grouped_range_values)
        values = {'__range_group_'+str(i): float(val) for i, val in
                enumerate(values)}
        value_type = 'range_grouped'
    else:
        values = value_fn(
                range_values['lower_bound'],
                range_values['upper_bound'],
                range_values['num_values'])
        values = [float(v) for v in values]
        value_type = 'range_ungrouped'
    return values, value_type

# Return a list of all the combinations resulting from adding each of the
# values to each of the current combinations except for those belonging to
# incompatible groups
def add_values_to_combinations(config_name, values, value_type, combinations):
    if value_type in ['list', 'range_ungrouped']:
        groups_by_value = [(val, None, set()) for val in values]
    elif value_type in ['grouped', 'range_grouped']:
        groups_by_value = [(val, group, set([g for g in values.keys() if not
            group == g])) for group, val in values.items()]
    new_combinations = []
    for value, group, excluded_groups in groups_by_value:
        for combination in combinations:
            if not group or group not in combination['excluded_groups']:
                new_combination = copy.deepcopy(combination)
                new_combination['excluded_groups'] |= excluded_groups
                new_combination['config_values'][config_name] = value
                new_combinations.append(new_combination)
    return new_combinations

def make_config_from_combination(combination, config_name_to_keys):
    changed_config = {}
    for config_name, value in combination['config_values'].items():
        config_keys = config_name_to_keys[config_name]
        section = changed_config
        for key in config_keys[:-1]:
            section[key] = {}
            section = section[key]
        section[config_keys[-1]] = value
    return changed_config

def make_configurations(base_config, changing_configurations, settings):
    
    # List of all the combinations of changing config options
    # Start with the trivial combination of no options included
    changing_config_combinations = [{
        'excluded_groups': set(),
        'config_values': {} # items are config_name: value
        }]
    # Also store the list of config keys corresponding to each config name
    config_name_to_keys = {}

    # List the combinations
    for config_name, config_settings in changing_configurations.items():
        if config_settings['value_type'] == 'range':
            values, value_type = parse_range_values(config_settings['values'],
                    settings['num_grouped_range_values'])
        elif config_settings['value_type'] in ['list', 'grouped']:
            values = config_settings['values']
            value_type = config_settings['value_type']
        changing_config_combinations = add_values_to_combinations(config_name,
            values, value_type, changing_config_combinations)
        config_name_to_keys[config_name] = config_settings['config']

    # Construct a dictionary of combinations and list of configurations
    combinations = {}
    configurations = []
    base_model_dir = base_config['Logging']['model_directory']
    for run_num, combination in enumerate(changing_config_combinations):
        run_name = 'run' + str(run_num).zfill(2)
        combinations[run_name] = combination['config_values']
        changed_config = make_config_from_combination(combination,
                config_name_to_keys)
        # Set the model directory to that corresponding to this run
        run_model_dir = os.path.join(base_model_dir, run_name)
        changed_config['Logging'] = {}
        changed_config['Logging']['model_directory'] = run_model_dir
        configurations.append((run_name, {**base_config, **changed_config}))
        
    return combinations, configurations

# Load config file containing all unchanging options as well as 
# Multiple Configurations sections
with open(args.config_path, 'r') as config_file:
    raw_config = yaml.load(config_file)
base_config = {key:val for key, val in raw_config.items() if key not in 
        ["Multiple Configurations Settings", "Multiple Configurations Values"]}
settings = raw_config["Multiple Configurations Settings"]
changing_configurations = raw_config["Multiple Configurations Values"]

# Generate a configuration for each combination of settings as specified
combinations, configurations = make_configurations(base_config,
        changing_configurations, settings)

# Save a dictionary storing the configuration combination set for each run
# to a file for convenient lookup
with open(settings['run_combinations_path'], 'w') as combinations_file:
    yaml.dump(combinations, combinations_file, default_flow_style=False)

# Run a model for each configuration combination
for run_name, config in configurations:
    print("Running", run_name+"...")
    #run_model(config, mode=args.mode, debug=args.debug,
    #        log_to_file=args.log_to_file)
