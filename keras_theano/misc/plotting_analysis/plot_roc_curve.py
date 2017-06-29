import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.ticker import MultipleLocator

#assumes gamma = 1, proton = 0

# parse command line arguments

parser = argparse.ArgumentParser(description='Predict on a batch of images and generate plots for the classifier value.')
parser.add_argument('--save_dir', help='directory to save plots in',default='.')
parser.add_argument('gamma_scores',help='text file containing list of classifier scores for gammas')
parser.add_argument('proton_scores',help='text file containing list of classifier scores for protons')
parser.add_argument('--filename',help='name of saved plot file',default='roc.png')

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
    gamma_pass = gamma[np.where( gamma > cut )]
    true_positive = gamma_pass.size

    gamma_fail = gamma[np.where( gamma < cut )]
    false_negative = gamma_fail.size

    true_positive_rate[i] = true_positive/(true_positive + false_negative)

    proton_pass = proton[np.where( proton > cut )]
    false_positive = proton_pass.size

    proton_fail = proton[np.where ( proton < cut )]
    true_negative = proton_fail.size

    false_positive_rate[i] = false_positive/(false_positive + true_negative)

## plot classifier value histograms

fig = plt.figure()
ax = plt.axes()
ax.plot(false_positive_rate,true_positive_rate)
ax.minorticks_on()
plt.minorticks_on()
ax.grid(True, which='both')

#plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

ml = MultipleLocator(5)
plt.axes().yaxis.set_minor_locator(ml)
plt.axes().xaxis.set_minor_locator(ml)

plotpath = os.path.join(save_dir,args.filename)
plt.savefig(plotpath)


