#!/bin/bash

#arguments
#1 -> full path of text file containing event numbers of gammas which passed the cuts
#2 -> full path of text file containing event numbers of protons which passed the cuts
#3 -> full path of directory containing gamma-diffuse images
#4 -> full path of directory containing proton images
#5 -> full path of target directory

cd $5
mkdir gamma-diffuse
mkdir proton

cd gamma-diffuse

for f in $(find $3 -name '*.png');
do
    filename=$(basename $f)
    id=$(echo $filename | cut -f1 -d_)

for i in $(awk -F "[ \t\n]+" 'FNR > 3 {print $4}' $1); 
do
    if [ "$id" == "$i" ];
    then
        ln -s $f $filename
        echo $filename
    fi

done

done

echo "Finished with gamma-diffuse"


cd ../proton

for f in $(find $4 -name '*.png');
do
    filename=$(basename $f)
    id=$(echo $filename | cut -f1 -d_)

for i in $(awk -F "[ \t\n]+" 'FNR > 3 {print $4}' $2); 
do
    if [ "$id" == "$i" ];
    then
        ln -s $f $filename
        echo $filename
    fi

done

done

echo "Finished with proton"





