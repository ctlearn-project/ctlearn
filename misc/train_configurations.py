"""
Conduct an automated hyperparameter search using the specified configuration
options.

train_configurations.py <constant_config_path> <changing_configs_path> <model_dir> <combinations_multiplier>
"""

import configparser
import itertools
import os
import sys

import numpy as np

from ctalearn.scripts.train import train

# Get command line arguments
constant_config_path = sys.argv[1]
changing_configs_path = sys.argv[2]
model_dir = sys.argv[3]
combinations_multiplier = int(sys.argv[4])

# Load a file specifying config options to run over
# Each line of the file has one of the formats:
# <section>, <key>, discrete: <value1>, <value2>,..., <valueN>
# <section>, <key>, grid_range: <lower_bound>, <upper_bound>, <spacing>, <num>
# <section>, <key>, random_range: <lower_bound>, <upper_bound>, <spacing>
# where <spacing> is one of "linear" or "log"
changing_configs = []
with open(changing_configs_path) as f:
    for line in f:
        if not line or line[0] == '#': continue
        config = {}
        config_def, settings = line.strip().split(':')
        config['section'], config['key'], iteration_type = [x.strip() for x in
                config_def.split(',')]
        settings = [s.strip() for s in settings.split(',')]
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

def yield_config(constant_config_path, changing_configs, model_dir,
        combinations_multiplier):
    # List tuples of the config coordinates and value for each value of each
    # discrete config
    discrete_configs = [[(c['section'], c['key'], v) for v in c['values']]
            for c in changing_configs if not c['random']]
    # Get all the combinations
    discrete_combinations = (list(itertools.product(*discrete_configs)) *
            combinations_multiplier)
    num_configs = len(list(discrete_combinations))
    # List tuples of the config coordinates and value for each value of each
    # random config
    random_configs = [[(c['section'], c['key'], v) for v in
        [c['value_fn']() for __ in range(num_configs)]] for
        c in changing_configs if c['random']]
    # Reorder to list by combinations
    random_combinations = list(zip(*random_configs))
    # Add the combinations together
    combinations = [d + r for d, r in zip(discrete_combinations,
        random_combinations)]
    for run_num, combination in enumerate(combinations):
        config = configparser.ConfigParser()
        config.read(constant_config_path)
        # Add the changing config options to the constant config
        for config_option in combination:
            section, key, value = config_option
            config[section][key] = str(value)
        # Set the model directory to that corresponding to this run
        run_model_directory = os.path.join(model_dir, 'run'+str(run_num))
        config['Logging']['ModelDirectory'] = run_model_directory
        yield config

for config in yield_config(constant_config_path, changing_configs, model_dir,
        combinations_multiplier):
    train(config, log_to_file=True)
