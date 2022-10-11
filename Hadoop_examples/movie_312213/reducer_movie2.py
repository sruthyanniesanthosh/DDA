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
    user_id, rating= line.split('\t')  
    
     #Converting string to float as we need the rating in float
    try:
        rating= float(rating)
    except ValueError:
        continue

    #Forming dictionary for each unique key, with values as list
    if user_id in dict: 
        dict[user_id].append(float(rating))
    else:
        dict[user_id] = []
        dict[user_id].append(float(rating))

#For each user in dictionary
#each userid is a key and its values contain a list of all 
# the ratings given by that user
for user in dict.keys():
    #If the number of ratings given by user is greater than or equal to 40
    if(len(dict[user])>=40):
        #Average rating calculation
        avg_rating = sum(dict[user])*1.0 / len(dict[user])
    
        #New dictionary with userid as key and the average rating as value
        new_dict[user]= avg_rating


min = 1.0
#Sorting dictionary in ascending order
sorted_keys = sorted(new_dict, key=new_dict.get)

for i in sorted_keys:
    #Finding the user with the least average rating
    if(new_dict[i] == min):
        #Forming the output as :userid  average rating
        string = '%s\t%s' %(i, new_dict[i])
        print(string)
    