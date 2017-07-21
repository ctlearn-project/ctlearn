import argparse
import numpy as np
from scipy.integrate import simps, trapz
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import glob
#import matplotlib.gridspec as gridspec

from matplotlib.ticker import MultipleLocator

#assumes gamma = 1, proton = 0

# parse command line arguments

parser = argparse.ArgumentParser(description='Combine ROC curves in single plot.')
parser.add_argument('--rocs', help='List of curves, as in /my/path/to/curves/*.txt. For the legend to work, filename should have the format: yourmodel_youbin.*',default='*.txt')
parser.add_argument('--save_dir', help='directory to save plots in',default='.')
parser.add_argument('--tag',help='name of saved plot file',default='roc_combined')

args = parser.parse_args()
myrocs = sorted(glob.glob(args.rocs))
nametag = args.tag
saveto = args.save_dir

# Create required directories
############################

save_dir = os.path.normcase(os.path.abspath(args.save_dir))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#gs = gridspec.GridSpec(2,1,height_ratios=[3, 1])

fs=12
dict_color = {'bin0': 'r', 'bin1': 'g', 'bin2': 'b'}
dict_style = {'ResNet50': ':', 'InceptionV3': '-', 'BDT': '-.'}
dict_marker = {'ResNet50': '.', 'InceptionV3': '', 'BDT': ''}
dict_alpha =  {'ResNet50': 1, 'InceptionV3': 1, 'BDT': 1}
dict_bins = {'bin0': 'Low energy', 'bin1': 'Medium energy', 'bin2': 'High energy'}

figx=6
figy=6
savefigs=True

for myroc in myrocs:
        model=myroc.split("/")[-1].split("-")[0]
        ebin=myroc.split("/")[-1].split("-")[1]
        leg="%s, %s" % (dict_bins[ebin],model)
        print(model, ebin)
        x, y = np.loadtxt(myroc)
        xnonzero = x[np.nonzero(x)]
        xnonzero = np.flipud(xnonzero)
        ynonzero = y[np.nonzero(x)]
        ynonzero = np.flipud(ynonzero)
        area = trapz(ynonzero,xnonzero)
        print("area (trapz) = %.5f" % (area))
        plt.figure(1,figsize=(figx,figy))
        plt.plot(x,y,color=dict_color[ebin],ls=dict_style[model],label=leg,marker=dict_marker[model],alpha=dict_alpha[model],markevery=0.1)
        leg1 = plt.legend(loc='lower right', shadow=False,fontsize=fs-2)

l = np.linspace(0, 1, 100)
plt.plot(x,x,color='black',ls=':',lw=0.5)
plt.axis([-0.05, 1, 0, 1.05])
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel('False Positive Rate',fontsize=fs)
plt.ylabel('True Positive Rate',fontsize=fs)

if savefigs:
    plt.savefig('%s/%s.eps' % (saveto,nametag), bbox_inches='tight')
    plt.savefig('%s/%s.png' % (saveto,nametag), bbox_inches='tight')
    plt.savefig('%s/%s.pdf' % (saveto,nametag), bbox_inches='tight')
plt.show()

