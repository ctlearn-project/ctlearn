import argparse
from itertools import cycle

from matplotlib import pyplot as plt
import pandas as pd
import sklearn.metrics

parser = argparse.ArgumentParser(
    description=("Plot ROC curves."))
parser.add_argument(
    'predictions_file',
    help='pandas hdf file of predictions')
parser.add_argument(
    "--output_filename",
    help="name for output plot file",
    default="roc_curves.png")
args = parser.parse_args()

gamma_classifier_values = pd.read_hdf(args.predictions_file, key='gamma')['reco_gammaness'].astype(float)
gamma_true_values = np.ones(len(gamma_classifier_values))
proton_classifier_values = pd.read_hdf(args.predictions_file, key='proton')['reco_gammaness'].astype(float)
proton_true_values = np.zeros(len(proton_classifier_values))

# Make the plot
plt.figure()

# Plot the ROC curve
classifier_values = np.concatenate((gamma_classifier_values, proton_classifier_values))
true_values = np.concatenate((gamma_true_values, proton_true_values))

fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_values, classifier_values)
auc = sklearn.metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, lw=2, label=classifier_name+'AUC = {:.2f}'.format(auc))

# Finish the plot
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')

plt.legend(loc='lower right')

plt.savefig(args.output_filename, bbox_inches='tight')
