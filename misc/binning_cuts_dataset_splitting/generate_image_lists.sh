#!/bin/bash

#arguments
#1 -> directory with gamma-diffuse images
#2 -> directory with proton images

for f in $(find $1 -name '*.png');
do
    echo $f >> gamma-diffuse.txt

done


for f in $(find $2 -name '*.png');
do
    echo $f >> proton.txt

done


