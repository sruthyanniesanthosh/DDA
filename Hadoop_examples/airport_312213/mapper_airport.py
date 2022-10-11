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
        dep_airport = line[3]  
        dep_delay = line[6]

        #If column value is empty, assigning 0
        if dep_delay == '':
            dep_delay = 0
            
        #Forming the output of mapper using tab character
        string = '%s\t%s' % (dep_airport, dep_delay)
        print(string)