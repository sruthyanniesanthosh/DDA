#!/usr/bin/env python
  

import sys

#Initialization
current_word = None
current_count = 0
word = None
  
# read the entire line from STDIN, which is the output of the mapper
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # splitting based on tab character as we wrote in mapper file
    word, count = line.split('\t')
    # convert count to int
    try:
        count = int(count)
    except ValueError:
        # count was not a number, discard
        continue
  
    # Hadoop sorts the map output by keys , ie, word before passing to reducer
    # Hence all same words will be together
    if current_word == word:
        #If the current word is same as new word, add count
        current_count += count
    else:
        if current_word:
            #end of one word, hence writing its count
            # write result to STDOUT
            #Current word and its count is overwrittern after each word is over
            print('%s\t%s' % (current_word, current_count))
        #Take the new word as current words, and its count as current count
        current_count = count
        current_word = word
  
# Last word 
if current_word == word:
    print('%s\t%s' % (current_word, current_count))