import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

#note that scores must be normalized (0 = proton, 1 = gamma)

# parse command line arguments

parser = argparse.ArgumentParser(description='Predict on a batch of images and generate plots for the classifier value.')
parser.add_argument('BDT_gamma_scores',help='text file containing list of BDT classifier scores for gammas')
parser.add_argument('BDT_proton_scores',help='text file containing list of BDT classifier scores for protons')
parser.add_argument('NN_gamma_scores',help='text file containing list of NN classifier scores for gammas')
parser.add_argument('NN_proton_scores',help='text file containing list of NN classifier scores for protons')
parser.add_argument('--save_dir',default='.',help='directory to save plot in')
parser.add_argument('--filename',default='plot.png',help='filename of plot')

args = parser.parse_args()

# Create required directories
############################

save_dir = os.path.normcase(os.path.abspath(args.save_dir))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


f, axarr = plt.subplots(2, sharex=True)

bins = np.linspace(0, 1, 100)

for method in ['BDT','NN']:

    #load data
    if method=='BDT':
        gamma = np.loadtxt(args.BDT_gamma_scores)
        proton = np.loadtxt(args.BDT_proton_scores)
        label = 'BDT'
        n = 0

    elif method=='NN':
        gamma = np.loadtxt(args.NN_gamma_scores)
        proton = np.loadtxt(args.NN_proton_scores)
        label = 'NN'
        n = 1 

    ## plot classifier value histograms

    axarr[n].hist(gamma, bins, alpha=0.5,histtype='stepfilled',label='gamma')
    axarr[n].hist(proton, bins, alpha=0.5,histtype='stepfilled',label='proton')
    axarr[n].set_title(label)

    axarr[n].set_xlabel('Classifier value')
    axarr[n].set_ylabel('Frequency')
    axarr[n].legend()

f.tight_layout()

plotpath = os.path.join(save_dir,args.filename)
plt.savefig(plotpath)
