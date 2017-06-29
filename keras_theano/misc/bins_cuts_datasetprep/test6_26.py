import os
import sys

with open(sys.argv[1],'r') as data:
    lines_after_4=data.readlines()[4:]
    for line in lines_after_4:
        if line[27]==" ":
            event=line[28:34]
        else:
            if line[26]==" ":
                event=line[27:34]
            else:
                event=line[26:34]
        if line[18]==" ":
            run= line[19:22]
        else: 
            if line[17]==" ":
                run=line[18:22]
            else:
                run=line[17:22]
        event_run=str(event+'_'+run)
        print(event_run)
