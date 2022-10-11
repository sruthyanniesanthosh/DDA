#Importing the required libraries
from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data                                                         


#Getting the rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

#Start time
start = MPI.Wtime()

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

""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1]):
        self.data = data
        self.partitions = []
        
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

#Calling Data partioner
batch_size = 128 / float(size)
partition_sizes = [1.0 / size for _ in range(size)]
partition = DataPartitioner(dataset_mnist, partition_sizes)
partition = partition.use(rank)

#Loading the MNIST dataset on the loader
train_loader = torch.utils.data.DataLoader(dataset = partition,
                                           batch_size = int(batch_size),
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = dataset_mnist_test,
                                           batch_size = 50,
                                           shuffle = True)


""" Function for weight averaging. """
def average_gradients(model):
    size = comm.Get_size()
    for param in model.parameters():
# Take the average of the weights of all workers to get new model params
        comm.allreduce(param.grad.data, op=MPI.SUM)
        param.grad.data /= size

#Initialize the model
model = Net()

#Setting the loss function
cost = nn.CrossEntropyLoss()

#Setting the optimiser
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

running_loss = 0.0
correct = 0
total = 0
total_step = len(train_loader)


#Training the model
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

        #Compute the average of the weights
        average_gradients(model)
        optimizer.step()

        #To find average of losses
        running_loss += loss.item()
        #Finding the predicted values and calculating accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        #writing loss and accuracy after the training for each worker       
print ('Epoch {} Training Loss: {:.4f} for learning rate: {} for worker : {}' 
                        .format(epoch+1,  running_loss/total_step ,learning_rate,rank))
print('Accuracy of the network on the train images: {:.4f}% for epoch: {} for worker: {}'.format(100 * correct / total,epoch+1,rank))
             


# Testing the model
# While testing, we don't compute gradients 
if(rank==0):
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
    print("\n Time taken for {} workers-{}".format(size,round(MPI.Wtime()-start,4)))
