import os
import sys

with open(sys.argv[1],'r') as f:
    for line in f:
        print(line)
        print("~~~~~~~~~~~")
        with open(sys.argv[2],'r') as s:
            for line in s:
               print(line)
               print("%%%%$$%%%")
