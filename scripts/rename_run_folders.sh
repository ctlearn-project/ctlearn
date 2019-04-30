#!/bin/bash

# Run this script 'bash rename_run_folders.sh' in the logging folder that contains the run folders.
# This script will rename the run folders automatically. Or you can set a path to the ctlearn/scripts
# folder in the config files ('Logging' section), which will rename the run folders by default.

if [ $# -ne 0 ]
then
    cd $1
fi
number_of_combinations=$(cat run_combinations.yml | grep '# Multiple configurations: ' | awk -vRS=")" -vFS="(" '{print $2}')
run_number=0
echo "Renaming the run folders:"
while [ $run_number -ne $number_of_combinations ]
do
    setting_value=$(cat run_combinations.yml | grep "# ($run_number) " | awk -vRS="]" -vFS="[" '{print $2}')
    if [ $run_number -lt 10 ]
    then
        echo run0$run_number $setting_value
        mv run0$run_number $setting_value
    else
        echo run$run_number $setting_value
        mv run$run_number $setting_value
    fi
    (( run_number++ ))
done
