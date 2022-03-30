# [Deprecated]
# This script will automatically rewrite the relevant yml fields in all the benchmark to your specification
# Just modify the variables below to match your configuration
# and execute this script from the folder whose yml files you want to modify

import yaml
import io
import os

# MODIFY THESE VARIABLES ########################
# file_list: .txt file containing in each line a path to a .h5 dataset
file_list = '/home/jsevillamol/Documentos/datasample/sample_files.txt'

# model_directory: path to ctlearn/models
model_directory = '/home/jsevillamol/Documentos/ctlearn/models'

# prediction_file_path: path to a folder where to store predictions
prediction_file_path = '/home/jsevillamol/Documentos/output/predictions'

# model_directory_log = specify model directory to store TensorFlow checkpoints and summaries, a timestamped copy of the run configuration, and optionally a timestamped file with logging output.
model_directory_log = '/home/jsevillamol/Documentos/output/logs'

##################################################

for fn in os.listdir('.'):
	# ignore non .yml files 
	if fn[-4:] != '.yml': continue

	#open yml file
	with open(fn, 'r') as stream:
		config = yaml.load(stream)

	# modify configuration
	config['Data']['file_list'] = file_list
	config['Model']['model_directory'] = model_directory
	config['Prediction']['prediction_file_path'] = prediction_file_path + '/prediction_' + fn[:-4] + '.csv'
	config['Logging']['model_directory'] = model_directory_log + '/' + fn[:-4]
	try:
		config['Multiple Configurations Settings']['run_combinations_path'] = model_directory_log + '/' + fn[:-4]
	except KeyError:
		pass

	# overwrite the previous configuration
	with io.open(fn, 'w', encoding='utf8') as outfile:
		yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)

	print('file {} overwritten'.format(fn))
