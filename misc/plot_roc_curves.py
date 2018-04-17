from itertools import cycle

from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics

import argparse

parser = argparse.ArgumentParser(
    description=("Plot ROC curves."))
parser.add_argument('predictions_list_file',
        help='list of paths to prediction files with names')
parser.add_argument(
    "--output_filename",
    help="name for output plot file",
    default="roc_curves.png")
args = parser.parse_args()

# Predictions list has the format: classifer_name, classifier_path
classifiers = []
with open(args.predictions_list_file) as f:
    for line in f:
        name, path = line.split(',')
        classifiers.append([name.strip(), path.strip()])

# Make the plot
plt.figure()
colors = cycle(['darkorange', 'aqua', 'cornflowerblue', 'deeppink'])

# Plot the ROC curve for each set of predictions
for classifier, color in zip(classifiers, colors):
    classifier_name = classifier[0]
    predictions_path = classifier[1]
    predictions = np.genfromtxt(predictions_path, delimiter=',', skip_header=1)
    true_labels = predictions[:,1]
    classifier_values = predictions[:,3]

    fpr, tpr, _ = sklearn.metrics.roc_curve(true_labels, classifier_values)
    auc = sklearn.metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, color=color, lw=2,
            label=classifier_name+' (AUC = {:.2f})'.format(auc))

# Finish the plot
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')

plt.legend(loc='lower right')

plt.show()
#plt.savefig(args.output_filename, bbox_inches='tight')

