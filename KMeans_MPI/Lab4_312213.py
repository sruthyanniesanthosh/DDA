#Importing the required libraries
from mpi4py import MPI
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy


#Getting the rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

#Start time
start = MPI.Wtime()

#Initializing the number of clusters
K = 2
centroid = []

#Function to calculate the euclidean distance btw each point and centroids
#Assigning the cluster membership according to this measure
def euclidean(matrix,centroid):
    print("Type of the given data-", type(matrix))
    print("\nData Type of centroid" , type(centroid))
    matrix_ = matrix.A
    cluster_membership = []
    #for each datapoint 
    for j in range((matrix_.shape)[0]):
        dist = []
        #for each centroid
        for i in range(K):
            #Calculating the euclidean distance
            dist.append(np.sqrt(np.sum(np.square(centroid[i]-matrix_[j]))))
        #Assigning the cluster having min distance value
        cluster_membership.append(dist.index(min(dist)))
        
    print("Length values of the data-", matrix_.shape)
    print("Length of cluster membership array-",len(cluster_membership))
    #Returning the array having the cluster membership values for each data point
    return cluster_membership


#Function to calculate the euclidean distance for convergence criteria
def dist(old_centroid,centroid):
    
    sum = 0
    print("Old-", old_centroid)
    print("Newie-", centroid)
    #For each cluster
    for i in range(K):
        #Finding the euclidean distance btw the old and new centroid
        x=scipy.sparse.csr_matrix.sqrt(old_centroid[i].dot(centroid[i].T))

        #Finding sum of all the distance values
        sum = sum + x.data[0]
    print("Sum of Difference btw centroids-", sum)
    #Returning the sum of the distance btw each centroid
    return sum


#Function to implement K-Means clustering
def kmeans(centroid, matrix):
    #Call distance func 
    membership_array = euclidean(matrix,centroid)

    #Calculating the Local centroid values for given data split
    new_centroid = []
    #for each cluster
    for j in range(K):
        elements = []
        #for each data point
        for i in range(matrix.shape[0]):
            #checking if the data belongs to the cluster
            if(membership_array[i] == j):
                #appending the data belonging to the given cluster
                elements.append(matrix[i,:])
        
        #New centroid appending the mean of all elements belonging to given cluster
        new_centroid.extend(elements[1].mean(axis=0))
    #  New local centroids for each cluster 
    print("New local centroid-",new_centroid)
    #returing the new centroids and the cluster membership array
    return new_centroid, membership_array


#Master worker
if(rank == 0):
    #Fetching the data using sklearn
    newsgroups_data = fetch_20newsgroups(subset="all")
    print("Number of documents-",len(newsgroups_data.data))
    #Finding TF-IDF matrix of the given data using sklearn
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(newsgroups_data.data)
    
    print(matrix.shape)
    print("\n Tf idf vector-", matrix) 
    print(type(matrix))


    #For each cluster
    for i in range(K):
        #Finding the initial centroids as random data points in the matrix
        centroid.extend((matrix[np.random.randint(0,matrix.shape[0])]).todense())
    
    #Spliting the data for different processes
    splits = np.array_split(range(matrix.shape[0]),size)
    matrix = [matrix[split] for split in splits]

    print("Number of Splits-",len(splits))
    print("Initial centroids-",centroid)

#Other workers
else:
    matrix = None
    centroid = None

#Scattering the data splits to all workers
matrix = comm.scatter(matrix,root=0)


#Value for checking convergence
converged = False
counter = 0


#While convergence criteria is not met
while(not converged and counter<50):
    #Broadcasting the centroids to all workers
    centroid = comm.bcast(centroid,root=0)
    
    old_centroid = centroid #saving copy of the centroid
    
    #Saving the old membership values after the initial run
    if(counter!=0 and rank == 0):
        old_membership = membership_full
    
    #Calling the K means function to calculate local centroids 
    local_centroid, membership = kmeans(centroid,matrix)
    
    #Gathering all the local centroids and cluster memberships for data splits
    local_centroids = comm.gather(local_centroid,root = 0)
    membership_full = comm.gather(membership, root = 0)
    
    
    global_centroid = []
    clusters = []
    new_centroids = []
    
    #Finding global centroid
    #master worker 
    if(rank == 0):
        #for each cluster
        for i in range(K):  
            #Initializing to the shape of centroid
            sum = np.zeros((173451))
            #for each worker
            for k in range(size):
                #Getting the local centroids for the given cluster 
                x = local_centroids[k][i]
                y = (np.asarray(x)).flatten()
                #Adding all the corresponding centroids to find the mean
                sum += y 
           
           #Converting the new centroid to csr matrix form
           #dividing sum by size to get the mean centroid
            new_centroid = scipy.sparse.csr_matrix(sum/size) 
            new_centroids.extend(new_centroid) #appending new centroids
        
        #Replacing the old with the new
        centroid = new_centroids
        print("New centroid-\n", centroid)
        counter+=1
        
        #checking if old and new are same after first run
        if(counter>2):
            #Convergence criteria, if the centroids are very near each other, 
            # then sum of their differences would be very less
            if(dist(old_centroid,centroid)<=1):
                    converged = True
                    print("K means Clustering Converged!")
                    #End Time
                    print("Time taken for kmeans clustering for {} clusters and {} workers-{} "
                    .format(K,size,round(MPI.Wtime()-start,4)))
            else:
                    print("-----------Iterating Again-----------")
                    print("Iteration number:",counter)
        
        #maximum iterations
        if(counter>=50):
            #Can occur if initial centroids are not chosen right
            print("Not converged in 50 iterations hence stopping")
            print("\n Time taken -",round(MPI.Wtime()-start,4))
            break
