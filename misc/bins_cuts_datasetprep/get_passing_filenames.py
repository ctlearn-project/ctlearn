import sys
import os
import numpy as np

filenames = np.loadtxt(sys.argv[1], dtype=str)
filenames = filenames
paths = [os.path.split(f)[0] for f in filenames]
files = [os.path.split(f)[1] for f in filenames]
passing_events = np.loadtxt(sys.argv[2], dtype=int)
passing_events = set([str(e) for e in passing_events])

passing_filenames = [p+'/'+f for p, f in zip(paths, files) if f.split('_')[0] in passing_events]

np.savetxt("passing_"+sys.argv[2], passing_filenames, fmt='%s')
