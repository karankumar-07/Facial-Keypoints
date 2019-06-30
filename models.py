## TODO: define the convolutional neural network architecture

import torch
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
        
        # inspired from ResNet 
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace = True)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.maxpool1 = nn.MaxPool2d(3,3)
        self.bn2 = nn.BatchNorm2d(64)
                
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3)        
        self.maxpool2 = nn.MaxPool2d(3,3)
        self.bn4 = nn.BatchNorm2d(256)

        self.m = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(22 * 22 * 256, 256)
        self.fc2 = nn.Linear(256, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.maxpool1(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.maxpool2(self.conv4(x))))
        x = x.view(-1, 22 * 22 * 256)
        x = self.m(self.relu(self.fc1(x)))
        x = self.fc2(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
