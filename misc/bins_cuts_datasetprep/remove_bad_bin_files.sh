#!/bin/bash

# usage: ./removed_bad_bin_files.sh <directory> <prefixes>
# prefixes is list of prefixes of files to delete, one per line

directory=$1
filenames=$2

#files_counted=0
#lines_read=0
#while read line; do
#    lines_read=$((lines_read + 1))
#    more_files=$(ls -U -1 "$directory/$line"* 2> /dev/null | wc -l)
#    files_counted=$((files_counted + more_files))
#    if [ "$more_files" -gt 0 ]; then
#        echo "Counted $line on line $lines_read"
#    fi
#done < "$filenames"
#echo "Number of files counted: $files_counted"

files_removed=0
lines_read=0
while read line; do
    lines_read=$((lines_read + 1))
    rm "$directory/$line"* 2> /dev/null && ! echo "Removed $line on line $lines_read" && files_removed=$((files_removed + 1)) 
done < "$filenames"
echo "Number of files deleted: $files_removed"
