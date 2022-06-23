#!/usr/bin/env python

import sys

#For each line in input file
for line in sys.stdin:

    # Removing leading, trailing whitespace and quotes
    line = line.strip().replace('\"', '')  
     # getting each attribute by splitting using :: as it is a dat file
    line = line.split("::")  

    if len(line) >=2 :
         #Getting the required column values
        user_id = line[0]  
        rating = line[2]
        
        #Forming the output of mapper using tab character
        string = '%s\t%s' % (user_id, rating)
        print(string)