#Importing the required libraries
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data        
import time
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

#Start time
start = time.time()

# Initializing params
num_classes = 10
learning_rate = 0.01
num_epochs = 10

class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(80, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 80)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#Function to train the model
def train_fn(model,train_loader,cost,optimizer,rank):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()
    #For each epoch
    for epoch in range(num_epochs):
    #For each batch
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):  
            
            #Forward pass
            outputs = model(images)
            #Calculating the loss
            loss = cost(outputs, labels)
                
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #To find average of losses
            running_loss += loss.item()
            #Finding the predicted values and calculating accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #writing loss and accuracy       
    print ('Epoch {} Training Loss: {:.4f} for learning rate: {} for process: {}' 
                            .format(epoch+1,  running_loss/len(train_loader) ,learning_rate, rank))
    print('Accuracy of the network on the train images: {:.4f}% for epoch: {}  for process:{}'.format(100 * correct / total,epoch+1, rank))
                

#Execution starts here
if __name__== '__main__':
    #Initializing the number of processes
    num_process = 5
    processes = []

    #Initializing the model and sharing the model
    model = Net()
    model.share_memory()

    #Setting the loss function
    cost = nn.CrossEntropyLoss()

    #Setting the optimiser
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    #Loading the MNIST dataset to fit the model given
    dataset_mnist = torchvision.datasets.MNIST(root = './data',
                                            train = True,
                                            transform = transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                            download = True)
    dataset_mnist_test = torchvision.datasets.MNIST(root = './data',
                                            train = False,
                                            transform = transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
                                            download = True)
    #Loading test dataset to test loader
    test_loader = torch.utils.data.DataLoader(dataset = dataset_mnist_test,
                                            batch_size = 50,
                                            shuffle = True)

    #For each process
    for rank in range(num_process):
    #Loading the required partition of data to the dataloader
    #Uses Distributed Sampler to partition dataset according to number of processes
            data_loader = DataLoader(
                dataset=dataset_mnist,
                sampler=DistributedSampler(
                    dataset=dataset_mnist,
                    num_replicas=num_process,
                    rank=rank
                ),
                batch_size=32
            )
            #Calling the train function using pytorch multiprocessing library
            p = mp.Process(target=train_fn, args=(model, data_loader,cost,optimizer,rank))
            p.start()
            #Append the model params
            processes.append(p)
# Joining of the processes as per the HogWild algorithm
    for p in processes:
            p.join()

#Testing done by one of the process only
    #Gardient not computed
    with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            running_loss = 0.0
            
        #For all images in test loader
            for i, (images, labels) in enumerate(test_loader):  
                
                #Forward pass
                outputs = model(images)
                loss = cost(outputs, labels)
                    
                #Finding average loss and accuracy
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            # the test loss and accuracy 
            print ('Average test Loss: {:.4f} for learning rate 0.01' 
                                .format(running_loss/len(test_loader)))
            print('Accuracy of the network on the test images: {:.4f} % for learning rate 0.01'.format(100 * correct / total))

      #End time      
    print("\n Time taken for {} processes -{}".format(num_process,round(time.time()-start,4)))
