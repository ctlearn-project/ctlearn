"""
Generate a list of configuration combinations and run a model for each, for
example, to conduct a hyperparameter search or simply to automate training or
prediction for a set of models.

The first argument to this script is the path to a CTLearn configuration file
containing two additional sections, Multiple Configurations Settings and 
Multiple Configurations Values. The Values section contains a dictionary of the
config parameters to be changed during the multiple runs and their possible
values, which may be explictly listed or automatically generated; all
combinations of the provided config parameters are generated except for
grouped parameters that are only combined as specified. The Settings section
contains other parameters for this script not specific to a particular config
parameter. See the example configuration file for a full explanation of the
structure and allowed values of these sections.

Grouped config options associate specific values of a config option with
values of other options. Groups must stay associated among config options,
that is, if a group is included for one config option it must be included for
any other config option that includes any of the groups of the first. If not,
the behavior is undefined! However, having distinct sets of groups is fine.

The config options not in the Multiple Configurations sections form the base
configuration when running the model and are the same for all runs. Any config
options also set in Multiple Configurations Values will overwrite those in the
base configuration. The outputs for each run are saved in subdirectories
<Logging:model_directory>/run00, <Logging:model_directory>/run01, etc.

In case the run sequence is interrupted, an option is provided to resume from 
a particular run. The remaining arguments are the flags for run_model.py,
passed in for each run.

In addition to the model outputs, a YAML file is also saved that, for each 
run, records the value of each config option listed under Multiple
Configurations Values. This file can be conveniently read by humans and
machines without having to separately examine the separate config files in each
run directory.
"""

import argparse
import copy
from multiprocessing import Pool
import os
import sys
import pkg_resources
import itertools

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
parser.add_argument(
        '--resume_from_run',
        type=int,
        default=0,
        help="resume from the nth run instead of the beginning")

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
    if range_values['num_values'] is None: # Grouped mode
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
            if group is None or group not in combination['excluded_groups']:
                new_combination = copy.deepcopy(combination)
                new_combination['excluded_groups'] |= excluded_groups
                new_combination['config_values'][config_name] = value
                new_combinations.append(new_combination)
    return new_combinations

def merge_config_from_combination(config, combination, config_name_to_keys):
    for config_name, value in combination['config_values'].items():
        config_keys = config_name_to_keys[config_name]
        section = config
        for key in config_keys[:-1]:
            if key not in section:
                section[key] = {}
            section = section[key]
        section[config_keys[-1]] = value

def make_configurations(base_config, changing_configurations, settings):
    
    # List of all the combinations of changing config options
    # Start with the trivial combination of no options included
    changing_config_combinations = [{
        'excluded_groups': set(),
        'config_values': {} # items are config_name: value
        }]
    # Also store the list of config keys corresponding to each config name
    config_name_to_keys = {}

    # Parse the settings
    num_grouped_range_values = settings.get('num_grouped_range_values', 1)

    # List the combinations
    for config_name, config_settings in changing_configurations.items():
        if config_settings['value_type'] == 'range':
            values, value_type = parse_range_values(config_settings['values'],
                    num_grouped_range_values)
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
        # Store the combination
        combinations[run_name] = combination['config_values']
        # Now construct the total config from the base and changing configs
        config = copy.deepcopy(base_config)
        merge_config_from_combination(config, combination, config_name_to_keys)
        # Set the model directory to the subdirectory for this run
        run_model_dir = os.path.join(base_model_dir, run_name)
        if 'Logging' not in config: config['Logging'] = {}
        config['Logging']['model_directory'] = run_model_dir
        configurations.append((run_name, config))
        
    return combinations, configurations

# Load config file containing all unchanging options as well as 
# Multiple Configurations sections
with open(args.config_path, 'r') as config_file:
    raw_config = yaml.load(config_file)
base_config = {key:val for key, val in raw_config.items() if key not in 
        ["Multiple Configurations Settings", "Multiple Configurations Values"]}
settings = raw_config["Multiple Configurations Settings"]
changing_configurations = raw_config["Multiple Configurations Values"]

# Detect the possible configuration combinations of the "Multiple Configurations
# Settings" and store them in the right order. This information will be written
# in the run_combination.yaml file as a comment, so that the rename_run_folders.sh
# can easily parse the comments and rename the run folders automatically.
base_model_dir = base_config['Logging']['model_directory']
base_scripts_dir = base_config['Logging'].get('scripts_directory', '')
multiple_config_values_all = raw_config["Multiple Configurations Values"]
multiple_config_values = []
for key in multiple_config_values_all.keys():
    values = []
    for val in multiple_config_values_all[key]['values'].keys():
        values.append(val)
    if values not in multiple_config_values:
        multiple_config_values.append(values)
multiple_config_values = np.array(multiple_config_values)

configuration_combinations = []
command = "itertools.product("
for i in np.arange(multiple_config_values.shape[0],0,-1):
    if i-1 != 0:
        command += "multiple_config_values[{}],".format(i-1)
    else:
        command += "multiple_config_values[{}])".format(i-1)

for combination in eval(command):
    configuration_combinations.append(combination)
configuration_combinations = np.array(configuration_combinations)

# Generate a configuration for each combination of settings as specified
combinations, configurations = make_configurations(base_config,
        changing_configurations, settings)

# Save a dictionary storing the configuration combination set for each run
# to a file for convenient lookup
print(settings['run_combinations_path'])
with open(settings['run_combinations_path'], 'w+') as combinations_file:
    ctlearn_version=pkg_resources.get_distribution("ctlearn").version
    combinations_file.write('# The training was performed using CTLearn version {}.\n'.format(ctlearn_version))
    combinations_file.write('# Multiple configurations: ({})\n'.format(configuration_combinations.shape[0]))
    for combination in np.arange(0,configuration_combinations.shape[0]):
        combinations_file.write('# ({}) ['.format(combination))
        for val in configuration_combinations[combination]:
            combinations_file.write('{}'.format(val))
        combinations_file.write(']\n')
    yaml.dump(combinations, combinations_file, default_flow_style=False)

# Run a model for each configuration combination from args.start
for run_name, config in configurations[args.resume_from_run:]:

    print("Running", run_name+"...")
    # Run models as subprocesses to free GPU memory after each run
    with Pool(1) as p:
        p.apply(run_model, (config,), dict(mode=args.mode, debug=args.debug,
                log_to_file=args.log_to_file))

if base_scripts_dir != '':
    os.system("cp {}rename_run_folders.sh {}".format(base_scripts_dir,base_model_dir))
    os.system("bash {}rename_run_folders.sh '{}'".format(base_model_dir,base_model_dir))



