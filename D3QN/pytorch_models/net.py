import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.distributions import normal

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias,0)

class QNetwork(nn.Module):
    def __init__(self, num_obs, num_actions, hidden_dim, seed):
        super(QNetwork, self).__init__()
        #self.seed = torch.manual_seed(seed)

        self.hidden_dim = hidden_dim
        self.hidden_dim = 8*10*64

        self.conv1 = nn.Conv2d(in_channels=num_obs, out_channels=32, kernel_size=10, stride=8, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        self.value1 = nn.Linear(self.hidden_dim, 512, bias=True) 
        self.value2 = nn.Linear(512, 512, bias=True) 
        self.value3 = nn.Linear(512, 1, bias=True)

        self.adv1 = nn.Linear(self.hidden_dim, 512, bias=True) 
        self.adv2 = nn.Linear(512, 512, bias=True) 
        self.adv3 = nn.Linear(512, num_actions, bias=True)

        self.apply(weights_init_)

    def forward(self, obs):
        
        o1 = F.relu(self.conv1(obs))
        o1 = F.relu(self.conv2(o1))
        o1 = F.relu(self.conv3(o1))
        o1 = o1.view(-1, self.hidden_dim)
        
        v1 = F.relu(self.value1(o1))
        v1 = F.relu(self.value2(v1))
        V = self.value3(v1)
        
        a1 = F.relu(self.adv1(o1))
        a1 = F.relu(self.adv2(a1))

        #print("A1 : ",a1.shape)
        A = self.adv3(a1)
    
        
        Q = V + A - A.mean()

        
        return Q, A, V


class Network(nn.Module):
    def __init__(self, num_obs, num_actions, hidden_dim, seed):
        super(Network, self).__init__()

        #self.seed = torch.manual_seed(seed)

        self.hidden_dim = 8*10*64

        self.conv1 = nn.Conv2d(in_channels=num_obs, out_channels=32, kernel_size=10, stride=8, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
   
        self.value1 = nn.Linear(self.hidden_dim, 512) 
        self.value2 = nn.Linear(512, 512) 
        self.value3 = nn.Linear(512, 1)

        self.adv1 = nn.Linear(self.hidden_dim, 512) 
        self.adv2 = nn.Linear(512, 512) 
        self.adv3 = nn.Linear(512, num_actions)


        self.apply(weights_init_)

    def forward(self, obs):
        o1 = F.relu(self.conv1(obs))
        #print("conv1 : ",o1.shape)
        o1 = F.relu(self.conv2(o1))
        #print("conv2 : ",o1.shape)
        o1 = F.relu(self.conv3(o1))
        #print("conv3 : ",o1.shape)
        o1 = o1.view(-1, self.hidden_dim)
        #print("view : ",o1)

        v1 = F.relu(self.value1(o1))
        v1 = F.relu(self.value2(v1))
        V = self.value3(v1)
        
        a1 = F.relu(self.adv1(o1))
        a1 = F.relu(self.adv2(a1))

        A = self.adv3(a1)
        
        Q = V + A - torch.mean(A, dim=-1, keepdim=True)

        return Q, A, V
        

    