#!/bin/bash

#arguments
#1 -> full path of text file containing event numbers of gammas which passed the cuts
#2 -> full path of text file containing event numbers of protons which passed the cuts
#3 -> full path of directory containing gamma-diffuse images
#4 -> full path of directory containing proton images
#5 -> full path of target directory

for i in $(awk -F "[ \t\n]+" 'FNR > 3 {print $4}' $1); 
d
