import argparse 
import numpy as np
import matplotlib.pyplot as plt
from pylab import genfromtxt
import os
import glob

parser = argparse.ArgumentParser(description='Plot accuracy and loss evolution versus epoch from a collection of training log files.')
parser.add_argument('mylogs',help='log files to be plotted')
parser.add_argument('tag',help='tag for output filenames')

args = parser.parse_args()

mylistlogs = sorted(glob.glob(args.mylogs))
nametag = args.tag

colormap = plt.cm.rainbow
colors = [colormap(i) for i in np.linspace(0, 1,len(mylistlogs))]
figx=8
figy=6

index=0
for mylog in mylistlogs:
    model=mylog.split("/")[-1].split("_")[0]
    lr=float(mylog.split("/")[-1].split("_")[1].split(".log")[0])
    leg="%s, lr=%.4f" % (model, lr)
    print(model, lr)
    with open(mylog) as f:
        next(f)
        lines = f.readlines()
        epoch = [line.split(",")[0] for line in lines]
        acc = [line.split(",")[1] for line in lines]
        loss = [line.split(",")[3] for line in lines]
        acc_val = [line.split(",")[4] for line in lines]
        loss_val = [line.split(",")[6] for line in lines]
        plt.figure(1,figsize=(figx,figy))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(epoch,acc,color=colors[index],label=leg)
        plt.plot(epoch,acc_val,color=colors[index],ls='dotted')    
        plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', shadow=False)
        plt.figure(2,figsize=(figx,figy))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(epoch,loss,color=colors[index],label=leg)
        plt.plot(epoch,loss_val,color=colors[index],ls='dotted')
        plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', shadow=False)
        index=index+1

plt.figure(1).savefig('accuracy_%s.png' % nametag, bbox_inches='tight')
plt.figure(2).savefig('loss_%s.png' % nametag, bbox_inches='tight')
plt.show()
