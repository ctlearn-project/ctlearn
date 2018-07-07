"""
Conduct an automated hyperparameter search using the specified configuration
options.

train_configurations.py <constant_config_path> <changing_configs_path> <run_combinations_path> <model_dir> <combinations_multiplier>

<constant_config_path>: path to configuration file containing all options meant
to be the same for every run

<changing_configs_path>: path to file specifying options to change on each run
Each line of the file has one of the formats:
<section>, <key>, discrete: <value1>, <value2>,..., <valueN>
<section>, <key>, grid_range: <lower_bound>, <upper_bound>, <spacing>, <num>
<section>, <key>, random_range: <lower_bound>, <upper_bound>, <spacing>
where <spacing> is one of "linear" or "log". Lines starting with '#' are
skipped as comments.

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
import configparser
import itertools
import os
import sys

import numpy as np

from ctalearn.scripts.train import train

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
    discrete_configs = [[(c['section'], c['key'], v) for v in c['values']]
            for c in changing_configs if not c['random']]
    # Get all the combinations
    discrete_combinations = (list(itertools.product(*discrete_configs)) *
            combinations_multiplier)
    combinations = [d + tuple([(c['section'], c['key'], c['value_fn']()) for
        c in changing_configs if c['random']]) for d in discrete_combinations]
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
        yield config, combination

# Load a file specifying config options to run over
changing_configs = []
with open(args.changing_configs_path) as f:
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

for run_num, (config, combination) in enumerate(yield_config(
    args.constant_config_path, changing_configs, args.model_dir,
    args.combinations_multiplier)):
    print(f"Training run {run_num}...")
    # Save the hyperparameter combination for each run to a file
    with open(args.run_combinations_path, 'a') as runs_file:
        runs_file.write(f"Run {run_num}\n")
        for param in combination:
            runs_file.write(','.join(map(str, param))+'\n')
        runs_file.write("\n")
    run_model(config, mode='train', debug=args.debug,
            log_to_file=args.log_to_file)
