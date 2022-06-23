## Basic Parallel Vector Operations with MPI

##3) To do Parallel Matrix Operation using MPI using collective communication

##Importing the required libraries
from mpi4py import MPI
import numpy as np

#Getting the rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

#Initializing the size of the matrices 
size_vector= [10,100,1000]
print("Number of workers-",size)

#For each size
for k in range(len(size_vector)):
    #Staring the timer
    start_time = MPI.Wtime()

    #master worker
    if(rank == 0):
        #initializing the matrices
        matrix_A = np.random.random(size=(size_vector[k],size_vector[k]))
        matrix_B = np.random.random(size=(size_vector[k],size_vector[k]))
        matrix_C = np.zeros((size_vector[k],size_vector[k]))

    #iterating through each row in A 
    for i in range(size_vector[k]):
        #iterating through each column in B
        for j in range(size_vector[k]):
            row_split = None
            col_split = None

            #master worker
            if(rank == 0):
                #splitting the matrices according to corresponding row (i)
                # and column(j) for matrix multiplication
                row_split = np.array_split(matrix_A[i,:],size)
                col_split = np.array_split(matrix_B[:,j],size)

            #sending the different sub splits of required row and column to 
            # all the other workers
            row_elts = comm.scatter(row_split, root = 0)
            col_elts = comm.scatter(col_split, root = 0)

            #Doing the element wise multiplication of the row and col splits 
            # and returning them all back to master
            pdt = comm.gather(np.matmul(row_elts,col_elts),root = 0)

            #master worker
            if(rank == 0):
                #sums the element wise products to get the new elt at position 
                # i ,j ( ith row and j th col)
                matrix_C[i][j]=  np.sum(pdt)

    #master worker
    if(rank == 0):
        #printing the output
        print("\nMatrix A -", matrix_A)
        print("\nMatrix B - ",matrix_B)
        print("\nThe Product of two matrices for number of workers {} and size of matrix {} -  {}".format(size,size_vector[k],matrix_C))

    #stopping the timer
    end_time = MPI.Wtime()

    total_time = end_time - start_time

    #printing the total time taken
    print("Total time taken for matrix multiplication of size {} with workers {} is {}".format(size_vector[k],size,total_time))