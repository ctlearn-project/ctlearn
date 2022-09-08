import argparse
from itertools import cycle

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics

parser = argparse.ArgumentParser(
    description=("Plot ROC curves."))
parser.add_argument(
    '--signal_file',
    help='pandas hdf file of signal predictions')
parser.add_argument(
    '--background_file',
    help='pandas hdf file of background predictions')
parser.add_argument(
    '--column_name',
    help='name of the column to plot',
    default="gammaness")
parser.add_argument(
    "--output_filename",
    help="name for output plot file",
    default="roc_curves.png")
args = parser.parse_args()

with pd.HDFStore(args.signal_file, mode="r") as f:
    events = f["/dl2/reco"]
    signal_classifier_values = events[args.column_name]
    signal_true_values = np.ones(len(signal_classifier_values))

with pd.HDFStore(args.background_file, mode="r") as f:
    events = f["/dl2/reco"]
    background_classifier_values = events[args.column_name]
    background_true_values = np.zeros(len(background_classifier_values))

# Make the plot
plt.figure()

# Plot the ROC curve
classifier_values = np.concatenate((signal_classifier_values, background_classifier_values))
true_values = np.concatenate((signal_true_values, background_true_values))

fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_values, classifier_values)
auc = sklearn.metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, lw=2, label='AUC = {:.2f}'.format(auc))

# Finish the plot
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')

plt.legend(loc='lower right')

print(f"ROC curve saved in '{args.output_filename}'.")
plt.savefig(args.output_filename, bbox_inches='tight')
