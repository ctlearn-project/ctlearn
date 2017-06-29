#!/bin/bash

#arguments
#1 -> directory with .png images
#2 -> filename for text file containing filenames

for f in $(find $1 -name '*.png');
do
    echo $f >> $2

done


