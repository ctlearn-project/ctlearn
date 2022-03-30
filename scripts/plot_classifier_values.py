import argparse

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description=("Plot histogram of classifier values."))
parser.add_argument(
    'predictions_file',
    help='pandas hdf file of predictions')
parser.add_argument(
    "--output_filename",
    help="name for output plot file",
    default="classifier_histogram.png")
args = parser.parse_args()

gamma_classifier_values = pd.read_hdf(args.predictions_file, key='gamma')['reco_gammaness'].astype(float)
proton_classifier_values = pd.read_hdf(args.predictions_file, key='proton')['reco_gammaness'].astype(float)

# Make the plot
plt.figure()

# Plot the histograms for both classifier values
bins = np.linspace(0, 1, 100)
plt.hist(gamma_classifier_values, bins, alpha=0.5, label='Gamma')
plt.hist(proton_classifier_values, bins, alpha=0.5, label='Proton')

plt.xlabel('Classifier value')
plt.ylabel('Counts')
plt.title('Histogram of classifier values')

plt.legend(loc='upper center')

plt.savefig(args.output_filename, bbox_inches='tight')

