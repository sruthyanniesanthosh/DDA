#Importing the required libraries
from mpi4py import MPI
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import nltk
import os
from os import walk
from collections import Counter
import math



#Getting the rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

#Function to read the file
def read_file(path):
    print("Reading the data")
    f = []
    text_doc = []
    for (dirpath, dirnames, filenames) in walk(path):

        file_paths = [os.path.join(dirpath, f) for f in filenames]
        for path in file_paths:
            f = open(path,'rb')
            text_doc.append(''.join(map(str, f.readlines())))
       
    return text_doc

#Function to clean and tokenize data
def data_clean(data):
    print("cleaning data")

    stopword_set = set(stopwords.words('english'))
    text_data = []
    for i,text in enumerate(data):
        text = word_tokenize(text)
        text = [word for word in text if word.isalpha()]
        text = [t.lower() for t in text]
        text = [word for word in text if word not in stopword_set]
        text_data.append(text)
    
    return text_data


#Function to calculate the Term Frequency
def TF(data):
    
    count = [dict(Counter(token)) for token in data]
    tf_token = [dict(Counter(token)) for token in data]

    for doc in tf_token:
        token_num = len(doc)
        for token in doc:
            doc[token] = doc[token] / token_num 
    
    return tf_token,count

#Function to count the frequency of each token in the whole corpus
#This data is needed for calculating IDF
def token_doc(tf_doc):
    tokdoc_freq = {}
    for tf_token in tf_doc:
        for token in tf_token:
            if token not in tokdoc_freq.keys():
                tokdoc_freq[token] = 1
            else:
                tokdoc_freq[token] = tokdoc_freq[token] + 1
    return tokdoc_freq

#Function to calculate IDF
def idf(tok_list,token_doccnt):
    start = MPI.Wtime()
    idf_doccnt = {}
    for token in tok_list:
        idf_doccnt[token] = math.log(19998/token_doccnt[token])
    
    return idf_doccnt

#Function to calculate TF-IDF
def tf_idf(idf_token,tf_token):
    print("TFIDF")
    tfidf_token = tf_token
    for doc in tfidf_token:
        for token in doc:
            doc[token] = doc[token] * idf_token[token]
    
    return tfidf_token

#Intializing
token = []
tf_list = []
tf_count = []
idf_dict = {}
tf_idf_list = []

#Starting time of the program
start_time = MPI.Wtime()

#Master worker
if (rank == 0):

    path = "20_newsgroups"
    start_read = MPI.Wtime()
    file_data = read_file(path)
    end_read = MPI.Wtime()
    print("\n Time taken for reading file -", round(end_read-start_read,4))
    #print(file_data[0])
    print("The number of documents in the corpus- ",len(file_data))

    #Splitting the data to different workers
    data_split = np.array_split(file_data, size-1)
    print("\nThe number of splits-",len(data_split))

    
    #Starting time for the Clean and Tokenization process
    start_clean = MPI.Wtime()
    #for loop to send data splits to other workers and to recieve the tokenized data back
    for i in range(size-1):
        comm.send(data_split[i-1],dest = i+1, tag=1)
        partial_token = comm.recv(source = MPI.ANY_SOURCE, tag =2)
        token.extend(partial_token)
    #End time for clean and tokenization process
    end_clean = MPI.Wtime()
    time_clean = end_clean - start_clean
    print("\n Total time taken for clean and tokenization- ",round(time_clean,4))
    #print("\n Number of documents- ",len(token))

   
    #Splitting the tokens to send to different workers
    token_split = np.array_split(token,size-1)
    
    #Starting time for TF calculation
    start_tf = MPI.Wtime()
    #For loop to send the tokens and to recieve the TF values from other workers
    for i in range(size-1):
        comm.send(token_split[i-1],dest = i+1, tag =3)
        tf_partial, count = comm.recv(source= MPI.ANY_SOURCE, tag=4)
        tf_list.extend(tf_partial)
        tf_count.extend(count)

    #print("TF list-\n",len(tf_count))
    #End time for TF process
    end_tf = MPI.Wtime()
    time_tf = end_tf - start_tf
    print("\n Total time taken for calculating Term Frequency- ",round(time_tf,4))

    #To find the number of times each token appears in the whole corpus
    token_freq_doc = token_doc(tf_count)
    #print("\nToken_doc-",token_freq_doc)

    #Splitting the token frequency data to send different tokens for different workers
    idf_split = np.array_split(list(token_freq_doc.keys()),size-1)
    
    #Start Time for IDF calculation
    start_idf = MPI.Wtime()
    #For loop for sending and recieving tokens and idf values 
    for i in range(size-1):
        comm.send(idf_split[i-1],dest = i+1, tag = 5)
        comm.send(token_freq_doc, dest = i+1, tag = 7)
        idf_val = comm.recv(source = MPI.ANY_SOURCE, tag = 6)
        idf_dict.update(idf_val)
    
    #End time for IDF process
    end_idf = MPI.Wtime()
    time_idf = end_idf - start_idf
    print("\n Total time taken for IDF calculation- ",round(time_idf,4))
    #print("IDF-", idf_dict)

    #Splitting the tf list to calculate TF_IDF values
    tf_split = np.array_split(tf_list,size-1)

    #Start time for TF_IDF process
    start_tfidf = MPI.Wtime()
    #For loop to send the data to other workers for getting TF-IDF values
    for i in range(size-1):
        comm.send(tf_split[i-1],dest = i+1, tag = 8)
        comm.send(idf_dict, dest =i+1, tag = 9)
        tf_idf_part = comm.recv(source= MPI.ANY_SOURCE, tag = 10)
        tf_idf_list.extend(tf_idf_part)

    print("TF-IDF-\n", tf_idf_list)
    #End time for TF_IDF process
    end_tfidf = MPI.Wtime()
    time_tfidf = end_tfidf - start_tfidf
    print("\n Total time taken for finding TFIDF- ",round(time_tfidf,4))
    
    #Total end for program
    end_time = MPI.Wtime()
    total_time = end_time - start_time 
    print("\n Total time taken for the whole program - ", round(total_time,4))



#For other workers
else:
    #splits are recieved and functions are called to perform various operations

    #Performing the operations for Cleaning and Tokenization 
    data = comm.recv(source=0, tag=1) #Data is recieved
    print("\nFile  has reached rank ",rank)
    #The required function is called and then result is send back to master
    comm.send(data_clean(data), dest = 0, tag =2)

    #Data recieved for TF calculation
    token_part = comm.recv(source=0, tag =3)
    #The required function is called and result is send back to master
    comm.send(TF(token_part), dest = 0, tag = 4)

    #Data recieved for IDF calcualation
    frq_part = comm.recv(source=0, tag = 5)
    frq_list = comm.recv(source=0, tag = 7)
    #The required function is called and result is send back to master
    comm.send(idf(frq_part,frq_list), dest = 0, tag = 6)

    #Data is recieved for TF-IDF calculation
    tf_spl = comm.recv(source =0, tag = 8)
    idf_spl = comm.recv(source = 0, tag = 9)
    #The required function is called and result is send back to master
    comm.send(tf_idf(idf_spl,tf_spl), dest = 0, tag = 10)


