#!/bin/bash

#arguments
#1 -> list text file containing event numbers of events which failed the cuts (gamma.txt)


while read i;
do 
    rm *$i*;
done < <(awk -F "[ \t\n]+" 'FNR > 3 {print $4}' $1) 




