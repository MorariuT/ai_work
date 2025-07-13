  
import torch
import torch.nn as nn
import torch.optim as optim
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 20, 3)
        self.conv3 = nn.Conv2d(20, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 40, 5)
        self.conv5 = nn.Conv2d(40, 64, 3)
        self.conv6 = nn.Conv2d(64, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        

        self.fc1 = nn.Linear(128, 70)
        self.fc2 = nn.Linear(70, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = (torch.relu(self.conv1(x)))
        x = (torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = (torch.relu(self.conv4(x)))
        x = (torch.relu(self.conv5(x)))
        x = self.pool(torch.relu(self.conv6(x)))
        # print(x.shape)  # Debug line to see the shape
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

