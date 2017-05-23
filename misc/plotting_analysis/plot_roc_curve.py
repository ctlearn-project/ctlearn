import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

#assumes gamma = 1, proton = 0

# parse command line arguments

parser = argparse.ArgumentParser(description='Predict on a batch of images and generate plots for the classifier value.')
parser.add_argument('save_dir', help='directory to save plots in')
parser.add_argument('gamma_scores',help='text file containing list of classifier scores for gammas')
parser.add_argument('proton_scores',help='text file containing list of classifier scores for protons')

args = parser.parse_args()

# Create required directories
############################

save_dir = os.path.normcase(os.path.abspath(args.save_dir))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#load data
gamma = np.loadtxt(args.gamma_scores)
proton = np.loadtxt(args.proton_scores)

#range of cuts
num_cuts = 1000
min_classifier_value = 0
max_classifier_value = 1
cuts= np.linspace(min_classifier_value,max_classifier_value,num_cuts)

#true positive and false positive

true_positive_rate = np.empty([num_cuts])
false_positive_rate = np.empty([num_cuts])

for i in range(0,num_cuts):
    cut = cuts[i]
    gamma_pass = np.where( gamma > cut )
    true_positive = gamma_pass.size

    gamma_fail = np.where( gamma < cut )
    false_negative = gamma_fail.size

    true_positive_rate[i] = true_positive/(true_positive + false_negative)

    proton_pass = np.where( proton > cut )
    false_positive = proton_pass.size

    proton_fail = np.where ( proton < cut )
    true_negative = proton_fail.size

    false_positive_rate[i] = false_positive/(false_positive + true_negative)

## plot classifier value histograms

plt.plot(false_positive_rate,true_positive_rate)
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plotpath = os.path.join(save_dir,'roc.png')
plt.savefig(plotpath)


