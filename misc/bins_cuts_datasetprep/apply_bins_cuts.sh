#!/bin/bash

#arguments
#1 -> full path of text file containing event numbers of gammas which passed the cuts
#2 -> full path of text file containing event numbers of protons which passed the cuts
#3 -> full path of directory containing gamma-diffuse images
#4 -> full path of directory containing proton images
#5 -> full path of target directory

#NOTE: text files should be in format "gamma-diffuse_1.txt" and "proton_1.txt" etc.
#NOTE: apply_cuts.sh script must be present in the same directory as this script
#NOTE: text files must be present in the same directory as this script
#
#arguments
#1 -> number of energy bins (starting from 0)
#2 -> full path of directory containing gamma-diffuse images
#3 -> full path of directory containing proton images
#4 -> full path of target directory (will create directories for each energy bin, eg. 0, 1,...)

end=$(expr $1 - 1)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SCRIPT=$DIR"/apply_cuts.sh"

cd $4

for i in $(seq 0 $end);
    do
        list1=$DIR"/gamma-diffuse_"$i".txt"
        list2=$DIR"/proton_"$i".txt"
        targetdir=$4"/"$i
        mkdir $i
        $SCRIPT "$list1" "$list2" "$2" "$3" "$targetdir"
    done


