import sys
import os

text_file= open('gamma_proton.txt','w')
for x in range (0,80):
    x_int=str(x)
    text_file.write('gamma-diffuse_'+x_int+'.txt\n')
    text_file.write('proton_'+x_int+'.txt\n')
text_file.close()
