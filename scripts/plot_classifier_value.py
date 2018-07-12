import argparse

from matplotlib import pyplot as plt
import numpy as np

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

predictions = np.genfromtxt(args.predictions_file, delimiter=',', names=True)
gamma_classifier_values = predictions['gamma']
proton_classifier_values = predictions['proton']

bins = np.linspace(0, 1, 100)

plt.hist(gamma_classifier_values, bins, alpha=0.5, label='Gamma')
plt.hist(proton_classifier_values, bins, alpha=0.5, label='Proton')

plt.xlabel('Classifier value')
plt.ylabel('Counts')
plt.title('Histogram of classifier values')

plt.legend(loc='upper center')

plt.savefig(args.output_filename, bbox_inches='tight')

