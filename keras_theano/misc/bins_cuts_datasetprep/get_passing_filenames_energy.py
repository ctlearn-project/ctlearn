import sys
import os
import numpy as np

#arg 1--> text file with list of all, unsorted files
#arg 2--> text file with list of all gamma-diffuse_*.txt or proton_*.txt fiels

filenames = np.loadtxt(sys.argv[1], dtype=str)
filenames = filenames
paths = [os.path.split(f)[0] for f in filenames]
files = [os.path.split(f)[1] for f in filenames]
with open(sys.argv[2],'r') as event_list:
    for line_1 in event_list:
        line_1 = line_1[:-1]
        events_list=open(line_1,'r')
        #goes line by line through gamma-diffuse/proton (run and event info)
        with events_list as data:
            lines_after_4=data.readlines()[3:]
            passing_filenames=[]
            passing_events=[]
            for line in lines_after_4:
                if line[27]==" ":
                    event = line[28:34]
                else:
                    if line[26]==" ":
                        event=line[27:34]
                    else:
                        event= line[26:34]
                if line[18]==" ":
                    run=line[19:22]
                else:
                    if line[17]==" ":
                        run=line[18:22]
                    else:
                        run=line[17:22]
                event_run=str(event+'_'+run)
                passing_events.append(event_run)
            passing_events=set(passing_events)
        for f,p in zip(files,paths):
            compare= f.split('_')[0]+'_'+f.split('_')[1]
            if compare in passing_events:    
                filename=str(p+'/'+f)
                filename=filename[2:-1]
                passing_filenames.append(filename)
        np.savetxt("passing_"+line_1, passing_filenames, fmt='%s')
