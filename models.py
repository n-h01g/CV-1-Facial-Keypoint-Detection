## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # convolutional layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 4, stride = 1)
        
        # convolutional layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1)
        
        # convolutional layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 2, stride = 1)
        
        # convolutional layer 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size = 1, stride = 1)
        
        # max pooling layer
        self.pool = nn.MaxPool2d(kernel_size = 2)
        
        # dense layer 1
        self.fc1 = nn.Linear(43264, 1024)
        
        # dense layer 2
        self.fc2 = nn.Linear(1024, 512)
        
        # dense layer 3
        self.fc3 = nn.Linear(512, 136)
        
        # dropout layer 1
        self.dropout1 = nn.Dropout(p = 0.10)

        # dropout layer 2
        self.dropout2 = nn.Dropout(p = 0.20)

        # dropout layer 3
        self.dropout3 = nn.Dropout(p = 0.30)

        # dropout layer 4
        self.dropout4 = nn.Dropout(p = 0.40)

        # dropout layer 5
        self.dropout5 = nn.Dropout(p = 0.50)

        # dropout layer 6
        self.dropout6 = nn.Dropout(p = 0.60)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # sequence of convolutional and max pooling layers
        # convolutional layer 1 with Exponential Linear Units (elu) activation function and dropout 1
        x = self.dropout1(self.pool(F.relu(self.conv1(x))))
        
        # convolutional layer 2 with Exponential Linear Units (elu) activation function and dropout 2
        x = self.dropout2(self.pool(F.relu(self.conv2(x))))
        
        # convolutional layer 3 with Exponential Linear Units (elu) activation function and dropout 3
        x = self.dropout3(self.pool(F.relu(self.conv3(x))))
        
        # convolutional layer 4 with Exponential Linear Units (elu) activation function and dropout 4
        x = self.dropout4(self.pool(F.relu(self.conv4(x))))
        
        # flatten image input
        x = x.view(x.size(0), -1)
        
        # dense layer 1 with Exponential Linear Units (elu) activation function
        x = self.dropout5(F.relu(self.fc1(x)))
        
        # dense layer 2 with Linear activation function
        x = self.dropout6(F.relu(self.fc2(x)))

        # dense layer 3
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
