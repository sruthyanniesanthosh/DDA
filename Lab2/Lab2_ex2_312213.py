## Basic Parallel Vector Operations with MPI

##2) To do Parallel Matrix Vector multiplication using MPI

 ##Importing the required libraries
from mpi4py import MPI
import numpy as np

#Getting the rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

#Initializing the size of the vector and matrix 
size_vector= [100,1000,10000]
#print("Number of workers-",size)

for j in range(len(size_vector)):

    #Master worker
    if(rank == 0):

        #Starting the timer
        start_time = MPI.Wtime()

        #Initializing the matrix and the vector
        matrix = np.random.random(size = (size_vector[j],size_vector[j]))
        vector = np.random.random(size_vector[j])
        print("\nShape of the matrix- ",matrix.shape)
        print("\nMatrix-", matrix)
        print("\nVector-", vector)
        C = []
        
        #Splitting the matrix horizontally 
        matrix_split = np.array_split(matrix, size-1)

        print("\nThe number of splits-",len(matrix_split))

        #for loop to send vector and matrix splits to other workers
        for i in range(size-1):       
            data = [matrix_split[i-1],vector]
            comm.send(data,i+1)

        #for loop to recieve the product of vector and matrix from other workers   
        for i in range(size-1):           
            final = comm.recv(source=MPI.ANY_SOURCE)
            C =np.append(C,final) #Appending all the partial products

        print("\nProduct of the vector and matrix of size {} by workers {} - {}".format(size_vector[j],size,C))
        print("\n Shape of final product-",C.shape)  

        #end timer
        end_time = MPI.Wtime()

        total_time = end_time - start_time  

        print("\nTime taken by {} workers for vector of size {} is {}".format(size,size_vector[j],total_time))

        ## Other workers
    else:
        #Vector and Matrix splits are recieved and products are found
        data = comm.recv(source=0)
        print("\nVector  has reached rank ",rank)
        print("\nMatrix split has reached rank ",rank)

        #Matrix multiplication is carried out(element wise multiplication of row and vector)
        pdt = np.matmul(data[0] , data[1])
            
        #The partial product is send back to master worker
        comm.send(pdt,0)

            