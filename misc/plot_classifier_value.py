from matplotlib import pyplot as plt
import numpy as np

import argparse

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

predictions = np.genfromtxt ('file.csv', delimiter=",")
true_labels = csv[:,1]
classifier_values = csv[:,4]

signal_classifier_values = classifier_values[true_labels==1]
background_classifier_values = classifier_values[true_labels==0]

bins = numpy.linspace(0, 1, 100)

plt.hist(signal_classifier_values, bins, alpha=0.5, label='Signal')
plt.hist(background_classifier_values, bins, alpha=0.5, label='Background')

plt.xlabel('Classifier value')
plt.ylabel('Counts')
plt.title('Histogram of classifier values')

plt.legend(loc='upper middle')

plt.savefig(args.output_filename, bbox_inches='tight')

