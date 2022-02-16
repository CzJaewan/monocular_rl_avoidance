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
    def __init__(self, num_obs, num_actions, hidden_dim):
        self.conv1 = nn.Conv1d(in_channels=num_obs, out_channel=32, kernel_size=(10,14), stride=8, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channel=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channel=64, kernel_size=3, stride=1, padding=1)
        
        self.value1 = nn.Linear(hidden_dim, 512) 
        self.value2 = nn.Linear(512, 1)

        self.adv1 = nn.Linear(hidden_dim, 512) 
        self.adv2 = nn.Linear(512, num_actions)

        self.apply(weights_init_)

    def forward(self, obs, action):
        o1 = F.relu(self.conv1(obs))
        o1 = F.relu(self.conv2(o1))
        o1 = F.relu(self.conv3(o1))
        o1 = o1.view(self.hidden_dim, -1)
        #o1 = o1.view(-1, self.hidden_dim)
        
        v1 = F.relu(self.value1(o1))
        value = F.relu(self.value2(v1))
        
        a1 = F.relu(self.adv1(o1))
        adv = F.relu(self.adv2(a1))
        
        # Q = value + (adv - advAvg)
        advAvg = torch.usqueeze(torch.mean(adv, 1), 1)
        advIdentifiable = torch.subtract(adv, advAvg)
        self.readout = torch.add(value, advIdentifiable)
        '''
        # define the ob cost function
		self.a = tf.placeholder("float", [None, ACTIONS])
		self.y = tf.placeholder("float", [None])
		self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), axis=1)
		
        #self.td_error = tf.square(self.y - self.readout_action)
        self.td_error = (self.y - self.readout_action)**2
        self.cost = tourch.mean(self.td_error)
        self.train_step = 
		self.cost = tf.reduce_mean(self.td_error)
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cost)
        '''
        action = self.readout[0]
        print("readout",readout.shape)
        return action #, self.readout

    def advantage(self, obs, action):

        o1 = F.relu(self.conv1(obs))
        o1 = F.relu(self.conv2(o1))
        o1 = F.relu(self.conv3(o1))
        o1 = o1.view(self.hidden_dim, -1)
        #o1 = o1.view(-1, self.hidden_dim)

        a1 = F.relu(self.adv1(o1))
        adv = F.relu(self.adv2(a1))
        
        return adv


class Network(nn.Module):
    def __init__(self, num_obs, num_actions, hidden_dim):
        self.conv1 = nn.Conv1d(in_channels=num_obs, out_channel=32, kernel_size=(10,14), stride=8, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channel=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channel=64, kernel_size=3, stride=1, padding=1)
        
        self.value1 = nn.Linear(hidden_dim, 512) 
        self.value2 = nn.Linear(512, 1)

        self.adv1 = nn.Linear(hidden_dim, 512) 
        self.adv2 = nn.Linear(512, num_actions)

        self.apply(weights_init_)

    def forward(self, obs, action):
        o1 = F.relu(self.conv1(obs))
        o1 = F.relu(self.conv2(o1))
        o1 = F.relu(self.conv3(o1))
        o1 = o1.view(self.hidden_dim, -1)
        #o1 = o1.view(-1, self.hidden_dim)
        
        v1 = F.relu(self.value1(o1))
        value = F.relu(self.value2(v1))
        
        a1 = F.relu(self.adv1(o1))
        adv = F.relu(self.adv2(a1))
        
        # Q = value + (adv - advAvg)
        advAvg = torch.usqueeze(torch.mean(adv, 1), 1)
        advIdentifiable = torch.subtract(adv, advAvg)
        self.readout = torch.add(value, advIdentifiable)
        
        '''
        # define the ob cost function
		self.a = tf.placeholder("float", [None, ACTIONS])
		self.y = tf.placeholder("float", [None])
		self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), axis=1)
		
        #self.td_error = tf.square(self.y - self.readout_action)
        self.td_error = (self.y - self.readout_action)**2
        self.cost = tourch.mean(self.td_error)
        self.train_step = 
		self.cost = tf.reduce_mean(self.td_error)
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cost)
        '''
        action = self.readout[0]
        
        print("readout",readout.shape)
        return action #, self.readout

    def advantage(self, obs, action):

        o1 = F.relu(self.conv1(obs))
        o1 = F.relu(self.conv2(o1))
        o1 = F.relu(self.conv3(o1))
        o1 = o1.view(self.hidden_dim, -1)
        #o1 = o1.view(-1, self.hidden_dim)

        a1 = F.relu(self.adv1(o1))
        adv = F.relu(self.adv2(a1))
        
        return adv

    