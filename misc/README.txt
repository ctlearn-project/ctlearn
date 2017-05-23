#############
#  misc     #
#############

Contains scripts to apply energy bins/other cuts using the ROOT files, to split the dataset into training/validation/test for Keras training, and to visualize using histograms and ROC curves.

instructions:

1. run getBDTscores.C on a ROOT file containing trained BDT results
OR
1. run predict_scores_inceptionv3.py with trained weights (checkpoint) on desired test data

2. use two output text files containing gamma and proton classifier values with plot_roc_curve.py or plot_classifier_curve.py

#############################
#  /misc/plotting_analysis  #
#############################

contains:

Makefile
########
use as is to compile calculateAccuracy.C, getBDTscores.C, plotImgPars.C.
make clean to clear all compiled files
make all to make all executables

getBDTscores.C
############
reads a BDT root file and writes the classifier values into 2 text files (corresponding to the true gamma and true proton events)

predict_scores_inceptionv3.py
#############################

takes a set of trained weights (saved checkpoint), loads them into an inceptionv3 network, then uses the network to predict (output classifier values) to 2 text files

NOTE: requires specific directory structure to work - see comments
NOTE: hardcoded to output into text files "predict_gamma.txt" and "predict_proton.txt"

plot_roc_curve.py
#################

takes two text files containing classifier values (normalized from 0 to 1), one for gamma and one for proton, and uses them to calculate and plot roc curves

use output from getBDTscores and predict_scores_inceptionv3.py to run

plot_classifier_curve.py
########################
takes two text files containing classifier values (normalized from 0 to 1), one for gamma and one for proton, and uses them to calculate and plot distribution histograms for the classifier scores

use output from getBDTscores and predict_scores_inceptionv3.py to run

plotImgPars.C
#############

reads ROOT file of event data and plots distribution histograms for a variety of parameters for visualization. 

see code for details


#################################
#  /misc/bins_cuts_datasetprep  #
#################################

contains scripts for applying energy binning, pre-training cuts, and splitting into train/validation/test directories for Keras training.

instructions for prepping dataset:

1. locate two directories, one containing all gamma images, one containing all proton images (it is fine if the images are in further subdirectories within)
2. run generate_image_lists.py on the two directories to generate two lists of all image filenames (gamma-diffuse.txt,proton.txt)
3. run generateBinLists to read mscw ROOT files (hardcoded), apply energy bins and other pre-training cuts (all hardcoded), and output to text files (ex. gamma-diffuse_0.txt, gamma-diffuse_1.txt, ... proton_0.txt, etc.). These text files list the eventIDs of events which pass the selection cuts and which fall into each energy bin.
4. pass the list of filenames and the list of passing eventIDs into get_passing_filenames.py to output the complete lists of image filenames which satisfy the cuts/bins (ex. passing_gamma-diffuse_1.txt, etc.)
5.run apply_bins_cuts_2.sh in the same directory as the lists of passing filenames and provide a target directory, where directories will be created (ex. /0, /1, /2, etc.) containing subdirectories (/gamma-diffuse, /proton) containing symlinks to the images satisfying the bins and cuts
6.run split_train_test.py to copy + split the data into the necessary training/validation/test directories for keras training. 

#####

img_lists
#########

contains pre-generated event lists for the sorting of the dataset. See instructions above.












#########
