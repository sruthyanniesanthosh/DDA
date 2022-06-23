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
    movie_id, movie_name, rating= line.split('\t')  
    
    #Converting string to float as we need the rating in float
    try:
        rating= float(rating)
    except ValueError:
        continue

    #Forming dictionary for each unique key, with values as list
    if movie_name in dict:  
        dict[movie_name].append(float(rating))
    else:
        dict[movie_name] = []
        dict[movie_name].append(float(rating))

#For each movie name
for name in dict.keys():
    #each movie name is a key and its values contain a list of all 
    # the ratings it got
    #Average rating calculation
    avg_rating = sum(dict[name])*1.0 / len(dict[name])
    
    #New dictionary with key as movie name and average rating as value
    new_dict[name]= avg_rating

max = 5.0

#Sorting the new dictionary is descending order
sorted_keys = sorted(new_dict, key=new_dict.get, reverse=True)

#for each movie name, if average rating is equal to the maximum rating,
#printing the same
for i in sorted_keys:
    if(new_dict[i] == max):
        #Forming output of the movie names with highest average rating
        string = '%s\t%s' %(i, new_dict[i])
        print(string)
    