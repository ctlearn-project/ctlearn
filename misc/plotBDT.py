import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

# parse command line arguments

parser = argparse.ArgumentParser(description='Predict on a batch of images and generate plots for the classifier value.')
parser.add_argument('save_dir', help='directory to save plots in')

args = parser.parse_args()

# Create required directories
############################

save_dir = os.path.normcase(os.path.abspath(args.save_dir))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#load data
gamma = np.loadtxt("BDT_gamma.txt")
proton = np.loadtxt("BDT_proton.txt")

#normalize to 0 to 1
gamma += 1
proton += 1

gamma /= 2
proton /=2

## plot classifier value histograms

histogram=plt.figure()
bins = np.linspace(0, 1, 100)

plt.hist(gamma, bins, alpha=0.5,histtype='stepfilled',label='gamma')
plt.hist(proton, bins, alpha=0.5,histtype='stepfilled',label='proton')
plt.legend()
plt.xlabel('BDT classifier value')
plt.ylabel('Frequency')
plotpath = os.path.join(save_dir,'BDT_plot.png')
plt.savefig(plotpath)
