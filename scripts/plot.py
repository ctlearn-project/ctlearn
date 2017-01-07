import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# parse command line arguments

parser = argparse.ArgumentParser(description='Generate accuracy and loss history plots from a Keras .csv log file')
parser.add_argument('log_file', help='path to .csv Keras log file')
parser.add_argument('--output_dir', help='directory for output plots (if used, specify absolute path)', default=os.getcwd())

args = parser.parse_args()

#get data from log file

data = np.genfromtxt(args.log_file, delimiter=',', skip_header=1, names=['epoch', 'acc', 'loss', 'val_acc', 'val_loss'])

run_name, ext = args.log_file.split(".", 1)

#plot accuracy history

plt.plot(data['epoch'],data['acc'])
plt.plot(data['epoch'],data['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='center right')

acc_plot_name ='accuracy[' + run_name + '].png'
plt.savefig(os.path.join(args.output_dir,acc_plot_name), bbox_inches='tight')

# plot loss history
plt.plot(data['epoch'],data['loss'])
plt.plot(data['epoch'],data['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='center right')

loss_plot_name ='loss[' + run_name + '].png'
plt.savefig(os.path.join(args.output_dir,loss_plot_name), bbox_inches='tight')


