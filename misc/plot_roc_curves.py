from matplotlib import pyplot as plt
import numpy as np
import sklearn

import argparse

parser = argparse.ArgumentParser(
    description=("Plot ROC curves."))
parser.add_argument(
    'predictions_file',
    help='csv file of predictions')
parser.add_argument(
    '--BDT_predictions_file',
    default=None)
parser.add_argument(
    "--output_filename",
    help="name for output plot file",
    default="roc_curves.png")
args = parser.parse_args()

predictions = np.genfromtxt ('file.csv', delimiter=",")
true_labels = csv[:,1]
classifier_values = csv[:,4]

fpr, tpr, _ = sklearn.metrics.roc_curve(true_labels, classifier_values, pos_label=1)
auc = sklearn.metrics.auc(fpr,tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='CNNRNN (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')

plt.legend(loc='lower right')

plt.savefig(args.output_filename, bbox_inches='tight')

