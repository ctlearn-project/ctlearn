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
fs=12
fsl=fs-2
savefigs=True
plot_val=True
plot_lr=False
plot_delta=False

index=0

dict_style = {'resnet50': '-', 'inceptionv3': '--'}
dict_color = {'bin0': 'r', 'bin1': 'g', 'bin2': 'b'}
dict_ebin = {'bin0': 'Low energy', 'bin1': 'Medium energy', 'bin2': 'High energy'}


gs = gridspec.GridSpec(2,1,height_ratios=[3, 1])

tw=3
vw=1

for mylog in mylistlogs:
    ebin=mylog.split("/")[-1].split("_")[0].split("-")[-3]
    model=mylog.split("/")[-1].split("_")[0].split("-")[-2]
    opti=mylog.split("/")[-1].split("_")[0].split("-")[-1]
    lr=float(mylog.split("/")[-1].split("_")[1].split(".log")[0])
    if plot_lr:
        leg="%s, %s, lr=%.4f" % (model, opti, lr)
    else:
        leg="%s, %s" % (model, dict_ebin[ebin])
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
        plt.tick_params(axis='both', which='major', labelsize=fs)

        if plot_val:
            plt.subplot(gs[0])
        plt.ylabel('Accuracy',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.plot(epoch,acc,color=dict_color[ebin],label=leg,ls=dict_style[model.lower()],lw=tw)
        if plot_val:
            plt.plot(epoch,acc_val,color=dict_color[ebin],ls=dict_style[model.lower()],lw=vw)
            #plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', shadow=False)
            plt.legend(loc='lower right', shadow=False,fontsize=fsl)
            plt.subplot(gs[1])
            plt.ylabel('$\Delta$Accuracy',fontsize=fs)
            plt.tick_params(axis='both', which='major', labelsize=fs)
            plt.plot(epoch,delta_acc,color=dict_color[ebin],label=leg,ls=dict_style[model.lower()])
        else:
            plt.legend(loc='lower right', shadow=False,fontsize=fsl)    
        plt.xlabel('Epoch',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)

        plt.figure(2,figsize=(figx,figy))
        if plot_val:
            plt.subplot(gs[0])
            plt.semilogy(epoch,loss_val,color=dict_color[ebin],ls=dict_style[model.lower()],lw=vw)
#            plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', shadow=False)
        
        plt.ylabel('Loss',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)

        plt.semilogy(epoch,loss,color=dict_color[ebin],label=leg,ls=dict_style[model.lower()],lw=tw)
        plt.legend(loc='upper right', shadow=False,fontsize=fsl)
        if plot_val:
            plt.subplot(gs[1])
            plt.ylabel('$\Delta$Loss',fontsize=fs)
            plt.tick_params(axis='both', which='major', labelsize=fs)

            plt.plot(epoch,delta_loss,color=dict_color[ebin],label=leg,ls=dict_style[model.lower()])
        else:
            plt.legend(loc='upper right', shadow=False,fontsize=fsl)      
        plt.xlabel('Epoch',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)

        index=index+1

if savefigs:
    plt.figure(1).savefig('accuracy_%s.eps' % nametag, bbox_inches='tight')
    plt.figure(2).savefig('loss_%s.eps' % nametag, bbox_inches='tight')
    plt.figure(1).savefig('accuracy_%s.png' % nametag, bbox_inches='tight')
    plt.figure(2).savefig('loss_%s.png' % nametag, bbox_inches='tight')
    plt.figure(1).savefig('accuracy_%s.pdf' % nametag, bbox_inches='tight')
    plt.figure(2).savefig('loss_%s.pdf' % nametag, bbox_inches='tight')
#plt.show()
