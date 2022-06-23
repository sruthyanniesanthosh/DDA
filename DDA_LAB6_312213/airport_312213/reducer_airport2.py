#!/usr/bin/env python
  

import sys
  
dict = {}
new_dict = {}
#for each line in the input of reducer 
# ( mapper output after sorting based on key)
for line in sys.stdin:
    #Removing trailing white spaces
    line = line.strip()
    #Getting the key and values send by mapper by splitting
    #according to tab character
    arr_airport, arr_delay= line.split('\t')  
    
    #Converting string to float as we need the arr delay in float
    try:
        arr_delay = float(arr_delay)
    except ValueError:
        continue

    #Forming dictionary for each unique key, with values as list
    if arr_airport in dict:  
        dict[arr_airport].append(float(arr_delay))
    else:
        dict[arr_airport] = []
        dict[arr_airport].append(float(arr_delay))

#For each arrival airport
for arr_airport in dict.keys():
     #each value in dict is a list of delays for that airport
    #average delay calc
    avg_delay = sum(dict[arr_airport])*1.0 / len(dict[arr_airport])

    #adding the average delay of each airport to a new dictionary
    new_dict[arr_airport]= avg_delay


count = 10

#Sorting in descending order based on the values
sorted_keys = sorted(new_dict, key=new_dict.get, reverse=True)

#Getting the 10 airports having highest average delay
for i in sorted_keys:
    #Forming output 
    string = '%s\t%s' %(i, new_dict[i])
    print(string)

    count = count-1
    if(count==0):
        break
