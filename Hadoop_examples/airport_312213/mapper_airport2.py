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
        arr_airport = line[4]  
        arr_delay = line[8]

        #If column value is empty, assigning 0
        if arr_delay == '':
            arr_delay = 0

        #Forming the output of mapper using tab character
        string = '%s\t%s' % (arr_airport, arr_delay)
        print(string)