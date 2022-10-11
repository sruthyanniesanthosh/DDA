 ## Basic Parallel Vector Operations with MPI

 ##1a) To find the sum of two vectors

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
        V2 = np.random.random(size_vector[j])
        print("\nThe length of the vectors - ",len(V1))
        print("\nThe first vector- ",V1)
        print("\nThe second vector- ",V2)
        V3 = []

        ##Splitting the vector so as to send to different workers
        V1_split = np.array_split(V1, size-1)
        V2_split = np.array_split(V2, size-1)

        print("\nThe number of splits-",len(V2_split))

        #for loop to send vector splits to other workers
        for i in range(size-1):       
            vector = [V1_split[i-1],V2_split[i-1]]
            comm.send(vector,i+1)

        #for loop to recieve the sum of vectors from other workers   
        for i in range(size-1):           
            final = comm.recv(source=MPI.ANY_SOURCE)
            V3 =np.append(V3,final) #Appending all the partial sums

        print("\nSum of the vectors of size {} by workers {} - {}".format(size_vector[j],size,V3))
        
        #stopping the timer
        end_time = MPI.Wtime()

        ##Finding the total time taken by parallelization
        total_time = end_time - start_time
        print("\nTime taken by {} workers for vector of size {} is {}".format(size,size_vector[j],total_time))



    ## Other workers
    else:
        #Vector splits are recieved and partial sums are found
        vector = comm.recv(source=0)
        print("\nVector 1 has reached rank ",rank)
        print("\nVector 2 has reached rank ",rank)
        sum = vector[0] + vector[1]
        
        #The partial sum is send back to master worker
        comm.send(sum,0)

        
        #https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Comm.html#mpi4py.MPI.Comm.send
        #https://rantahar.github.io/introduction-to-mpi/aio/index.html
        #https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        #https://stackoverflow.com/questions/64311489/siice-a-numpy-array-based-on-an-interval
        #https://stackoverflow.com/questions/40336601/python-appending-array-to-an-array
        #https://stackoverflow.com/questions/41575243/matrix-multiplication-using-mpi-scatter-and-mpi-gather
        #https://www.christianbaun.de/CGC1718/Skript/CloudPresentation_Shamima_Akhter.pdf