#!/usr/bin/env python

import sys

#Reading each line in the input file
for line in sys.stdin:
    line = line.strip() #Removing white space characters from beginning and end
    words = line.split() #Splitting based on space
    for word in words:
        #Sending each word and value 1 to reducer
        print('%s\t%s' % (word, 1))
