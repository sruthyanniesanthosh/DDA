 ## Basic Parallel Vector Operations with MPI

 ##1b) To find the average  of numbers in a vector

 ##Importing the required libraries
from mpi4py import MPI
import numpy as np

#Getting the rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

#Initializing the size of the vectors 
size_vector= [1000,10000,100000,3]
print("Number of workers-",size)

#for loop for each vector size
for j in range(len(size_vector)):


    #master worker
    if (rank==0):

        #Starting the timer
        start_time = MPI.Wtime()

        #Initializing the two vectors with random numbers
        V1 = np.random.random(size_vector[j])
        print("\nThe length of the vectors - ",len(V1))
        print("\nThe input vector:",V1)
        V2 = []

        ##Splitting the vector so as to send to different workers
        V1_split = np.array_split(V1, size-1)
        
        print("\nThe number of splits-",len(V1_split))

        #for loop to send vector splits to other workers
        for i in range(size-1):       
            vector = V1_split[i-1]
            comm.send(vector,i+1)


        #for loop to recieve the avg of vectors from other workers   
        for i in range(size-1):           
            final = comm.recv(source=MPI.ANY_SOURCE)
            V2 =np.append(V2,final) #Appending all the partial averages

        print("\nAverage of the vector of size {} by workers {} - {}".format(size_vector[j],size,np.mean(V2)))
    
        #stopping the timer
        end_time = MPI.Wtime()

        ##Finding the total time taken by parallelization
        total_time = end_time - start_time
        print("\nTime taken by {} workers for vector of size {} is {}".format(size,size_vector[j],total_time))


        
   
    ## Other workers
    else:
        #Vector splits are recieved and partial means are found
        vector = comm.recv(source=0)
        print("\nVector 1 has reached rank ",rank)
        
        avg = np.mean(vector)
        
        #The partial sum is send back to master worker
        comm.send(avg,0)

        