import argparse
from itertools import cycle

from matplotlib import pyplot as plt
import pandas as pd
import sklearn.metrics

parser = argparse.ArgumentParser(
    description=("Plot ROC curves."))
parser.add_argument('predictions_list_file',
        help='list of paths to predictions files with names')
parser.add_argument(
    "--output_filename",
    help="name for output plot file",
    default="roc_curves.png")
args = parser.parse_args()

# Predictions list has the format: classifer_name, classifier_path
classifiers = []
with open(args.predictions_list_file) as f:
    for line in f:
        if not line or line[0] == '#': continue
        name, path = line.split(',')
        classifiers.append([name.strip(), path.strip()])

# Make the plot
plt.figure()
colors = cycle(['darkorange', 'aqua', 'cornflowerblue', 'deeppink'])

# Plot the ROC curve for each set of predictions
for classifier, color in zip(classifiers, colors):
    classifier_name = classifier[0]
    predictions_path = classifier[1]
    predictions = pd.read_hdf(predictions_path)
    labels = predictions['mc_particle'].astype(int)
    gamma_classifier_values = predictions['reco_gammaness'].astype(float)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels,
            gamma_classifier_values, pos_label=1)
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

plt.savefig(args.output_filename, bbox_inches='tight')
#plt.show()
