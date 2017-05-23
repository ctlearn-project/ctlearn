import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

#note that scores must be normalized (0 = proton, 1 = gamma)

# parse command line arguments

parser = argparse.ArgumentParser(description='Predict on a batch of images and generate plots for the classifier value.')
parser.add_argument('gamma_scores',help='text file containing classifier scores for true gammas')
parser.add_argument('proton_scores',help='text file containing classifier scores for true protons')

args = parser.parse_args()

# Create required directories
############################

save_dir = os.path.normcase(os.path.abspath(args.save_dir))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#load data
gamma = np.loadtxt(args.gamma_scores)
proton = np.loadtxt(args.proton_scores)

## plot classifier value histograms

histogram=plt.figure()
bins = np.linspace(0, 1, 100)

plt.hist(gamma, bins, alpha=0.5,histtype='stepfilled',label='gamma')
plt.hist(proton, bins, alpha=0.5,histtype='stepfilled',label='proton')
plt.legend()
plt.xlabel('Classifier value')
plt.ylabel('Frequency')
plotpath = os.path.join(save_dir,'classifier_scores_histogram.png')
plt.savefig(plotpath)
