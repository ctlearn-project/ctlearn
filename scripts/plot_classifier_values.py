import argparse

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description=("Plot histogram of classifier values."))
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
    default="classifier_histogram.png")
args = parser.parse_args()

# Make the plot
plt.figure()

# Plot the histograms for both classifier values
bins = np.linspace(0, 1, 100)
if args.signal_file:
    with pd.HDFStore(args.signal_file, mode="r") as f:
        events = f["/dl2/reco"]
        signal_classifier_values = events[args.column_name]
        plt.hist(signal_classifier_values, bins, alpha=0.5, label='Signal')
if args.background_file:
    with pd.HDFStore(args.background_file, mode="r") as f:
        events = f["/dl2/reco"]
        background_classifier_values = events[args.column_name]
        plt.hist(background_classifier_values, bins, alpha=0.5, label='Bkg')

plt.xlabel(f'Classifier value ({args.column_name})')
plt.ylabel('Counts')
plt.title('Histogram of classifier values')

plt.legend(loc='upper center')

print(f"Histogram of classifier values saved in '{args.output_filename}'.")
plt.savefig(args.output_filename, bbox_inches='tight')

