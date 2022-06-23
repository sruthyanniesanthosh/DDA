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
    genre, rating= line.split('\t')  
    
    #Converting string to float as we need the rating in float
    try:
        rating= float(rating)
    except ValueError:
        continue

     #Forming dictionary for each unique key, with values as list
    if genre in dict:  
        dict[genre].append(float(rating))
    else:
        dict[genre] = []
        dict[genre].append(float(rating))

#For each genre categories corresponding to movies in dictionary
#each genre list is a key and its values contain a list of all 
# the ratings movies belonging to that genre list got
for genre in dict.keys():
    #Average rating calculation 
    avg_rating = sum(dict[genre])*1.0 / len(dict[genre])
 
 #New dictionary with genre listas key and the average rating as value
    new_dict[genre]= avg_rating

count = 1

##Sorting dictionary in descending order to get the highest average rating
sorted_keys = sorted(new_dict, key=new_dict.get, reverse=True)

for i in sorted_keys:
    if(count>0):
        #The genre list having the highest average rated movies is found
        #Output is formed as : Genre list  Average rating
        string = '%s\t%s' %(i, new_dict[i])
        print(string)
        count = count - 1
    