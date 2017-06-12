import sys
import os
import numpy as np

filenames = ['gamma-diffuse_incorrect_0.txt','proton_incorrect_0.txt','gamma-diffuse_incorrect_1.txt','proton_incorrect_1.txt','gamma-diffuse_incorrect_2.txt','proton_incorrect_2.txt']

current_dir = os.getcwd()

for filename in filenames:

    print("starting " + filename)

    out_filename = filename.split(".")[0] + '_filenames.txt'

    f_out = open(os.path.join(current_dir,out_filename),'w')

    with open(os.path.join(current_dir,filename),'r') as f_in:
        for line in f_in:
            if line[2] == '*':
                continue
            a,row,b,eventID,c,MCe0,d = line.split()
            if row.isdigit():
                MCe0_formatted = MCe0.split(".")[0] + '.' + MCe0.split(".")[1][:3]
                line_filename = eventID + '_' +  MCe0_formatted + 'TeV\n'
                f_out.write(line_filename)

    f_out.close()

    print("complete")




