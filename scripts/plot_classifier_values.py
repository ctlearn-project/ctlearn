import argparse

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description=("Plot histogram of classifier values."))
parser.add_argument(
    'predictions_file',
    help='csv file of predictions')
parser.add_argument(
    "--output_filename",
    help="name for output plot file",
    default="classifier_histogram.png")
args = parser.parse_args()

predictions = pd.read_hdf(args.predictions_file)
gamma_mask = predictions['mc_particle'] == 1
gamma_classifier_values = predictions['reco_gammaness'][gamma_mask]
proton_classifier_values = predictions['reco_gammaness'][~gamma_mask]

bins = np.linspace(0, 1, 100)

plt.hist(gamma_classifier_values, bins, alpha=0.5, label='Gamma')
plt.hist(proton_classifier_values, bins, alpha=0.5, label='Proton')

plt.xlabel('Classifier value')
plt.ylabel('Counts')
plt.title('Histogram of classifier values')

plt.legend(loc='upper center')

plt.savefig(args.output_filename, bbox_inches='tight')

