#Importing the required libraries
from mpi4py import MPI
import numpy as np
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
    file = []
    full_doc = []
    file_paths = []
    for (dirpath, dirnames, filenames) in walk(path):
        #getting the paths of all files using python lists
        file_paths = [os.path.join(dirpath, f) for f in filenames]
        
        #Reading the text per document
        for path in file_paths:
            file = open(path,'rb')
            x = map(str,file.readlines())#joining all the text in a doc
            full_doc.append(''.join(x))  #appending to an array
       
    return full_doc

#Function to clean and tokenize data
def data_clean(data):
    print("cleaning data")
    text_data = []
    doc_data = []
    #Getting the list of stop words from nltk
    stopwords_ = set(stopwords.words('english'))
    
    #For iterating over each document
    for i,doc in enumerate(data):
        #Getting each word in a doc
        text = word_tokenize(doc)
        #Converting to lower case
        text2 = [t.lower() for t in text]
        #Getting only the words ( removing all other characters)
        text3= [word for word in text2 if word.isalpha()]  
        #Removing stop words
        text4 = [word for word in text3 if word not in stopwords_]
            
        text_data.append(text4)
    
    return text_data


#Function to calculate the Term Frequency
def TF(data):
    #Finding the count of each token in a document using Counter and python lists
    tf_token = [dict(Counter(token)) for token in data]
    
    #TF is found within each document
    for doc in tf_token:
        #Total number of tokens in the doc
        token_len = len(doc)
        for token in doc:
# TF = number of times token appears in the doc / total no of tokens in doc
            doc[token] = doc[token] / token_len
    
    return tf_token


#Function to count the occurance tokens in the whole corpus
#This data is needed for calculating IDF
def token_count(data):
    #Counter to calculate the count of each token in each doc
    tok_freq = {} #dictionary
    freq_doc = [dict(Counter(token)) for token in data]
    
    for count_token in freq_doc:#for each doc
        for token in count_token: # for each token
            if token in tok_freq.keys(): # checking if token already exists
                tok_freq[token] += 1 #increment count
            else:
                tok_freq[token] = 1 #add new token
    return tok_freq

#Function to calculate IDF
def idf(tok_list,token_count):
    idf_corpus = {} #dictionary
    C = 19998 #Total number of documents
    #for each token in the given data

    for token in tok_list:
        #idf = total number of docs / count of the documents having the token
        idf_corpus[token] = math.log(C/token_count[token])
    
    return idf_corpus



#Function to calculate TF-IDF
def tf_idf(idf_dict,tf_list):
    print("TFIDF")
    #Iterating through each document in data
    for file in tf_list:
        #Iterating through each token
        for word in file:
            #TFIDF(token, doc) = tfvalue(token in the doc)* idfvalue(token)
            file[word] = file[word] * idf_dict[word]

    tf_idf = tf_list
    return tf_idf

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
    #Reading the file and storing the text per document in an array of lists.
    file_data = read_file(path)
    end_read = MPI.Wtime()
    print("\n Time taken for reading file -", round(end_read-start_read,4))
    #print(file_data[5])
    print("The number of documents in the corpus- ",len(file_data))

    #Splitting the data to different workers
    data_split = np.array_split(file_data, size-1)
    print("\nThe number of splits-",len(data_split))

    
    #Starting time for the Clean and Tokenization process
    start_clean = MPI.Wtime()
    #for loop to send data splits to other workers and to recieve the tokenized data back
    for i in range(size-1):
        comm.send(data_split[i-1],dest = i+1, tag=1)
        #Recieving tokens for the documents send
        partial_token = comm.recv(source = MPI.ANY_SOURCE, tag =2) 
        token.extend(partial_token) #appending the tokens
    #End time for clean and tokenization process
    end_clean = MPI.Wtime()
    time_clean = end_clean - start_clean
    print("\n Total time taken for clean and tokenization- ",round(time_clean,4))
    print("\n Tokenised data- ",token)

   
    #Splitting the tokens to send to different workers
    token_split = np.array_split(token,size-1)
    
    #Starting time for TF calculation
    start_tf = MPI.Wtime()
    #For loop to send the tokens and to recieve the TF values from other workers
    for i in range(size-1):
        comm.send(token_split[i-1],dest = i+1, tag =3)
        tf_partial = comm.recv(source= MPI.ANY_SOURCE, tag=4)
        tf_list.extend(tf_partial) #appending the values to get all the documents
    

    print("TF list-\n",tf_list)
    #End time for TF process
    end_tf = MPI.Wtime()
    time_tf = end_tf - start_tf
    print("\n Total time taken for calculating Term Frequency- ",round(time_tf,4))

    #To find the number of times each token appears in the whole corpus
    token_freq_doc = token_count(token)
    #print("\nToken_doc-",token_freq_doc)

    #Splitting the token frequency data to send different tokens for different workers
    idf_split = np.array_split(list(token_freq_doc.keys()),size-1)
    
    #Start Time for IDF calculation
    start_idf = MPI.Wtime()
    #For loop for sending and recieving tokens and idf values 
    for i in range(size-1):
        #sending the token split and the frequency of each token
        data = [idf_split[i-1],token_freq_doc]
        comm.send(data,dest = i+1, tag = 5)
        idf_val = comm.recv(source = MPI.ANY_SOURCE, tag = 6)
        idf_dict.update(idf_val) #updating in dictionary the idf value of each token
    
    #End time for IDF process
    end_idf = MPI.Wtime()
    time_idf = end_idf - start_idf
    print("\n Total time taken for IDF calculation- ",round(time_idf,4))
    print("\nIDF-", idf_dict)

    #Splitting the tf list to calculate TF_IDF values
    tf_split = np.array_split(tf_list,size-1)

    #Start time for TF_IDF process
    start_tfidf = MPI.Wtime()
    #For loop to send the data to other workers for getting TF-IDF values
    for i in range(size-1):
        #sending the termfrequency list split and the idf dictionary
        data = [tf_split[i-1],idf_dict]
        comm.send(data,dest = i+1, tag = 8)
        tf_idf_part = comm.recv(source= MPI.ANY_SOURCE, tag = 10)
        tf_idf_list.extend(tf_idf_part) #appending the partial values

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
    frq_part, frq_list = comm.recv(source=0, tag = 5)
    #The required function is called and result is send back to master
    comm.send(idf(frq_part,frq_list), dest = 0, tag = 6)

    #Data is recieved for TF-IDF calculation
    tf_spl, idf_spl = comm.recv(source =0, tag = 8)
    #The required function is called and result is send back to master
    comm.send(tf_idf(idf_spl,tf_spl), dest = 0, tag = 10)


