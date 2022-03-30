"""
[Deprecated]
Run this script from the folder where the run folders are situated.
It will automatically read the results and create a table summarizing them.
It currently has columns for telescope type, array or single tel, AUROC
"""

import os
import yaml
import re
import pandas as pd

rows = []

for folder in os.listdir('.'):
	if not os.path.isdir(folder): continue
	dir_ = os.listdir(folder)

	# open configuration file
	r = re.compile('.*config\.yml')
	matches = list(filter(r.match, dir_))
	assert(len(matches) == 1)
	fn = folder + '/' + matches[0]

	with open(fn, 'r') as stream:
		config = yaml.load(stream)

	# open log
	r = re.compile('.*logfile\.log')
	matches = list(filter(r.match, dir_))
	assert(len(matches) == 1)
	fn = folder + '/' + matches[0]

	with open(fn, 'r') as stream:
		r = re.compile('INFO:Saving dict for global step .*')
		matches = list(filter(r.match, stream))
		assert(len(matches) > 0)
		val_info = matches[-1]

	# compile the info we need
	input_type = config['Data']['mode']
	tel_type = config['Data']['selected_telescope_type']

	auroc = float(re.findall(r"auc = [-+]?\d*\.*\d+", val_info)[0][6:])

	rows.append({'input_type': input_type, 'tel_type': tel_type, 'auroc': auroc})

df = pd.DataFrame(rows, columns=['input_type', 'tel_type', 'auroc'])
df = df.sort_values(by=['input_type', 'tel_type'])

df.to_csv('summary.csv')
