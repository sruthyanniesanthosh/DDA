#!/usr/bin/env python

import sys
#For each line in input
for line in sys.stdin:

    # Removing leading, trailing whitespace and quotes
    line = line.strip().replace('\"', '')  
    # getting each attribute by splitting using , as it is a csv file
    line = line.split(",")  
    
    if len(line) >= 2:
        #Getting the required column values
        movie_id = line[0]  
        movie_name = line[1]
        rating = line[4]
        
        #Forming the output of mapper using tab character
        string = '%s\t%s\t%s' % (movie_id, movie_name,rating)
        print(string)