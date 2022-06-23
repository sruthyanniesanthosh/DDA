#!/usr/bin/env python
  
import sys
  
dict = {}

#for each line in the input of reducer 
# ( mapper output after sorting based on key)
for line in sys.stdin:
    #Removing trailing white spaces
    line = line.strip()
    #Getting the key and values send by mapper by splitting
    #according to tab character
    dep_airport, dep_delay= line.split('\t')  
    #print(line)
    #Converting string to float as we need the dep delay in float
    try:
        dep_delay = float(dep_delay)
    except ValueError:
        continue

    #Forming dictionary for each unique key, with values as list
    if dep_airport in dict: 
        dict[dep_airport].append(float(dep_delay))
    else:
        dict[dep_airport] = []
        dict[dep_airport].append(float(dep_delay))

#For each departure airport
for dep_airport in dict.keys():
    #each value in dict is a list of delays for that airport
    #average delay calc
    avg_delay = sum(dict[dep_airport])*1.0 / len(dict[dep_airport])
    #max delay
    max_delay = max(dict[dep_airport])
    #min delay
    min_delay = min(dict[dep_airport])

    #Forming output of reducer by using tab as separator
    #Format - Departure Airport, Average delay, Maximum delay, Minimum delay
    string = '%s\t%s\t%s\t%s' % (dep_airport, avg_delay,max_delay,min_delay)
    #printing to output
    print(string)
