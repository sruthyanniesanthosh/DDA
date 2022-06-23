#Importing the required libraries
from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Getting the rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

#Start time
start = MPI.Wtime()

#Function for preprocessing the data
def preprocessing(df_kdd):
    #replacing categorical values
    for col in df_kdd.columns:
            df_kdd[col] = df_kdd[col].astype('category')
            df_kdd[col] = df_kdd[col].cat.codes

    df_kdd = df_kdd.dropna()
    #Taking random sample
    df_kdd = df_kdd.sample(frac = 1, random_state=311)
    #Taking target y 
    df_kdd_x = df_kdd.drop(["TARGET_D"], axis=1)
    df_kdd_y = df_kdd[["TARGET_D"]]

    #Applying min-max normalization on x and y
    df_kdd_x=(df_kdd_x - df_kdd_x.min())/(df_kdd_x.max() - df_kdd_x.min())   
    df_kdd_y = (df_kdd_y - df_kdd_y.min()) / (df_kdd_y.max() - df_kdd_y.min())    
    

    print(df_kdd["TARGET_D"].value_counts()) 
    print(df_kdd["TARGET_D"].unique()) 
    '''''
    nan_values = df_kdd_x.isna()
    nan_columns = nan_values.any()

    columns_with_nan = df_kdd_x.columns[nan_columns].tolist()
    columns_with_nan
    '''''
    #Dropping unwanted target column, nan column
    df_kdd_x = df_kdd_x.drop(columns=['TARGET_B','CONTROLN','RFA_2R'], axis=1)

    return df_kdd_x, df_kdd_y

#Function to split data into 70% for training and 30 % for testing
def train_test_split(df_kdd_x,df_kdd_y):
    train_df_x = df_kdd_x.iloc[:int(95412*0.7) , :]#70% for training, 30% for testing
    test_df_x = df_kdd_x.drop(train_df_x.index)
    train_df_y = df_kdd_y.iloc[:int(95412*0.7) , :]#70% for training, 30% for testing
    test_df_y = df_kdd_y.drop(train_df_y.index)

    #Converting data frames to arrays
    y_train_df = np.array(train_df_y) # puttingthe target feature to y array
    x_train_df = np.array(train_df_x)

    y_test_df = np.array(test_df_y)
    x_test_df = np.array(test_df_x)


    print("Shape of x",x_train_df.shape)
    print("Shape of y",y_train_df.shape)

    return x_train_df, y_train_df, x_test_df, y_test_df

#Function to calculate the rmse value
def rmse(y_predicted, y):
    error = y - y_predicted
    rmse = np.sqrt(np.mean(error ** 2))
    return rmse

#Function to perform stochastic gradient descent
def PSGD(x,y,beta,lr):
    #for each data point
    for i in range(x.shape[0]):
        #Finding y
        y_predicted = np.dot(x[i],beta)
       
        b = x[i]
        b=b[:,np.newaxis]
        diff = y[i] - y_predicted
        diff = diff[:,np.newaxis]
        #Finding gradient
        grad = -2 * np.dot(b,diff)
        #finding new beta
        new_beta = beta - lr * grad
        beta = new_beta

    return beta


#master worker
if (rank==0):
    #Reading data set
    df_kdd = pd.read_csv('kdd.txt',low_memory=False)
    #Pre processing data
    df_kdd_x,df_kdd_y = preprocessing(df_kdd)
    #Splitting data for training and testing
    x_train_df, y_train_df, x_test_df, y_test_df = train_test_split(df_kdd_x,df_kdd_y)

    #Splitting data for parallelism
    x_train_split = np.array_split(x_train_df,size)
    y_train_split = np.array_split(y_train_df, size)

#other workers
else:
    x_train_split = None
    y_train_split = None


#Initialization
rmse_train = []
rmse_test = []
lr = 0.0001

#Value for checking convergence
converged = False
max = 200
counter = 0


#scatter the divided train dataset across workers
x_train_split = comm.scatter(x_train_split,root=0)
y_train_split = comm.scatter(y_train_split,root=0)

#initializing beta 
beta = np.zeros((x_train_split.shape[1],1))
print("shape of beta-", beta.shape)

#While convergence criteria is not met
while(not converged and counter<max):
    #Broadcasting the beta to all workers
    beta = comm.bcast(beta,root=0)

    #master worker
    if(rank == 0):
        #Finding y train and y test values
        #Calculating the rmse values for y predicted and true
        y_predicted_train = np.dot(x_train_df,beta)
        rmse_train.append(rmse(y_predicted_train,y_train_df))
    
        y_predicted_test = np.dot(x_test_df,beta)
        rmse_test.append(rmse(y_predicted_test,y_test_df))

    #Calling the function to do stochastic gradient descent for each split
    local_beta = PSGD(x_train_split,y_train_split,beta,lr)
    comm.barrier()

    #collecting the betas for each split
    local_betas = comm.gather(local_beta,root = 0)
    
    #master worker
    if(rank == 0):
        print("local betas, ", len(local_betas))
        #Finding beta calculation
        beta = np.mean(local_betas, axis=0)

        if(counter>1):
            #storing rmse values for checking
            current_rmse = rmse_train[counter]
            prev_rmse = rmse_train[counter-1]
            
            #convergence criteria check
            if(prev_rmse - current_rmse < 0.000001):
                converged = True
                print("------Converged------")
                print("Iteration- ", counter)
                print("\n Time taken for convergence for {} workers-{}".format(size,round(MPI.Wtime()-start,4)))
                break
            
            else:
                print("------Continue------")
                print("Iteration - ", counter)

        #maximum iterations
        if(counter == max-1):
                #Can occur if learning rate are not chosen right
                print("Not converged in {} iterations hence stopping".format(max))
                print("\n Time taken for {} workers-{}".format(size,round(MPI.Wtime()-start,4)))
                break
        counter+=1

    if(rank == 0):
       #Writing the RMSE  results to csv
        df = pd.DataFrame({'Train RMSE':rmse_train,'Test RMSE': rmse_test})
        df.to_csv('kdd_results_7.csv')


