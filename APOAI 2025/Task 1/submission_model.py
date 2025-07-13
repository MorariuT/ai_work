  
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #######Please design your model here########
        self.hidden1 = nn.Linear(2, 6)
        self.ac1 = nn.ReLU()
        self.hidden2 = nn.Linear(6, 6)
        self.ac2 = nn.ReLU()
        self.output = nn.Linear(6, 1)
        self.ac3 = nn.Sigmoid()
        
    def forward(self, x):
        #######Please design your model here########
        x = self.ac1(self.hidden1(x))
        x = self.ac2(self.hidden2(x))
        x = self.ac3(self.output(x))
        return x


