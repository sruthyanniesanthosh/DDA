#!/usr/bin/env python

import sys

#For each line in input file
for line in sys.stdin:

    # Removing leading, trailing whitespace and quotes
    line = line.strip().replace('\"', '') 

    # getting each attribute by splitting using , as it is a csv file
    line = line.split(",") 

    if len(line) >=2 :
        #Getting the required column values
        genre = line[2]  
        rating = line[4]
        
        #Forming the output of mapper using tab character
        string = '%s\t%s' % (genre, rating)
        print(string)