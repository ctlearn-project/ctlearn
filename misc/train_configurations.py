"""
Conduct an automated hyperparameter search using the specified configuration
options.

train_configurations.py <n_samples> <constant_config_path> <changing_configs_path>
"""

import configparser
import subprocess
import sys

# Load a config file with options to be the same for every run
constant_config_path = sys.argv[1]
constant_config = configparser.ConfigParser()
constant_config.read(constant_config_path)

# Load a file specifying config options to run over
# Each line of the file has the format:
# <section>,<key>,<grid_type>,discrete:<value1>,<value2>,...,<valueN>
# or
# <section>,<key>,<grid_type>,range:<lower_bound>,<upper_bound>,<scale>
# where <grid_type> is one of "random" or "fixed" and <scale> is one of 
# "linear" or "log"
changing_configs = []
with open(changing_configs_path) as f:
    for line in f:
        if not line or line[0] == '#': continue
        config = {}
        config_def, setting_values = line.strip().split(':')
        section, key, grid_type, iteration_type = config_def.split(',')
        (config['section'], config['key'], config['grid_type'],
                config['iteration_type'] = config_def.split(','))
        config['setting_values'] = setting_values.split(',')
        changing_configs.append(config)

def yield_config(constant_config, changing_configs):

    yield config

for __ in n_samples:
    config = yield_config(constant_config, changing_configs)

