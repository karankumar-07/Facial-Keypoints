import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim
import tqdm

batch_size = 3	
data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])

transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv', root_dir='data/training/', transform=data_transform)
train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv', root_dir='data/test/')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,  num_workers=6)

def train_net(n_epochs):
    net.train()
    for epoch in range(n_epochs): 
        running_loss = 0.0
        for batch_i, data in tqdm.tqdm(enumerate(train_loader)):
            images = data['image']																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																
            key_pts = data['keypoints']

            images = images.to(device, dtype = torch.float)
            key_pts = key_pts.to(device, dtype = torch.float)

            optimizer.zero_grad()
            key_pts = key_pts.view(key_pts.size(0), -1)
	      
            output_pts = net(images)
            loss = criterion(output_pts, key_pts)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0

    print('Finished Training')

n_epochs = 10 
net = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters())
train_net(n_epochs)