import argparse 
import numpy as np
import matplotlib.pyplot as plt
from pylab import genfromtxt
import os
import glob
import matplotlib.gridspec as gridspec

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
savefigs=True
plot_val=False
plot_lr=False

index=0

dict_style = {'resnet50': '-', 'inceptionv3': '--'}

gs = gridspec.GridSpec(2,1,height_ratios=[3, 1])

for mylog in mylistlogs:
    model=mylog.split("/")[-1].split("_")[0].split("-")[-2]
    opti=mylog.split("/")[-1].split("_")[0].split("-")[-1]
    lr=float(mylog.split("/")[-1].split("_")[1].split(".log")[0])
    if plot_lr:
        leg="%s, %s, lr=%.4f" % (model, opti, lr)
    else:
        leg="%s, %s" % (model, opti)
#    leg="%s, lr=%.6f" % (model, lr)
    print(leg)
    with open(mylog) as f:
        next(f)
        lines = f.readlines()
        epoch = [float(line.split(",")[0]) for line in lines]
        acc = [float(line.split(",")[1]) for line in lines]
        loss = [float(line.split(",")[2]) for line in lines]
        acc_val = [float(line.split(",")[3]) for line in lines]
        loss_val = [float(line.split(",")[4]) for line in lines]
        delta_acc = [x - y for x, y in zip(acc, acc_val)]
        delta_loss = [x - y for x, y in zip(loss, loss_val)]
        plt.figure(1,figsize=(figx,figy))
        if plot_val:
            plt.subplot(gs[0])
        plt.ylabel('Accuracy')
        plt.plot(epoch,acc,color=colors[index],label=leg,ls=dict_style[model.lower()])
        if plot_val:
            plt.plot(epoch,acc_val,color=colors[index],ls='dotted')
            plt.subplot(gs[1])
            plt.ylabel('Delta Accuracy')
            plt.plot(epoch,delta_acc,color=colors[index],label=leg)
            plt.subplot(gs[0])
            plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', shadow=False)
        else:
            plt.legend(loc='lower right', shadow=False)    
        plt.xlabel('Epoch')
        
        plt.figure(2,figsize=(figx,figy))
        if plot_val:
            plt.subplot(gs[0])
            plt.semilogy(epoch,loss_val,color=colors[index],ls='dotted')
            plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', shadow=False)
        
        plt.ylabel('Loss')
        plt.semilogy(epoch,loss,color=colors[index],label=leg,ls=dict_style[model.lower()])
        if plot_val:
            plt.subplot(gs[1])
            plt.ylabel('Delta Loss')
            plt.plot(epoch,delta_loss,color=colors[index],label=leg)
        else:
            plt.legend(loc='upper right', shadow=False)      
        plt.xlabel('Epoch')
        index=index+1

if savefigs:
    plt.figure(1).savefig('accuracy_%s.eps' % nametag, bbox_inches='tight')
    plt.figure(2).savefig('loss_%s.eps' % nametag, bbox_inches='tight')
    plt.figure(1).savefig('accuracy_%s.png' % nametag, bbox_inches='tight')
    plt.figure(2).savefig('loss_%s.png' % nametag, bbox_inches='tight')
plt.show()
