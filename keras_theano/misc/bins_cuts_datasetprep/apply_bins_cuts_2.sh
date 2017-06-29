#!/bin/bash

#NOTE: text files should be in format "passing_gamma-diffuse_1.txt" and "passing_proton_1.txt" etc. # Changed to this format - AB
#NOTE: text files must be present in the same directory as this script
#
#arguments
#1 -> number of energy bins
#2 -> full path of target directory (will create sub-directories for each energy bin, eg. 0, 1,...)

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: "
    echo $0" <# of energy bins> <full path to target dir> <original path substitution> [optional]"
    echo "For path substitutions, input path to the 'img' folder only."
fi

end=$(expr $1 - 1)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $2

for i in $(seq 0 $end);
    do
        cd $2

        list1=$DIR"/passing_gamma-diffuse_"$i".txt"
        list2=$DIR"/passing_proton_"$i".txt"
        targetdir=$2"/"$i
        mkdir $i
        
        mkdir $i"/gamma-diffuse"
        cd $i"/gamma-diffuse"
        echo $i"/gamma-diffuse"
    
        while read j;
            do
                filename="${j##*/}"
		if [ $# -eq 3 ]; then
		    tmp=`echo $j | rev | cut -d "/" -f1-5 | rev`
		    j=$3"/"$tmp
		fi
		ln -s $j
                #echo $filename

        done < $list1

        cd "../.."
 
        mkdir $i"/proton"
        cd $i"/proton"
        echo $i"/proton"
    
        while read j;
            do
                filename="${j##*/}"
		if [ $# -eq 3 ]; then
		    tmp=`echo $j | rev | cut -d "/" -f1-5 |rev`
		    j=$3/$tmp
		fi
                ln -s $j
                #echo $filename

        done < $list2

    done


