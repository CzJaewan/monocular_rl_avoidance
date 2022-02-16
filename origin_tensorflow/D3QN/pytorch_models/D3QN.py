import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.optim import Adam
from pytorch_models.network import QNetwork, Network

class D3QN(object):
    def __init__(self, num_obs, num_action, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        
        self.epsilon = args.epsilon
        self.epilon_decay = args.epsilon_decay
        self.eps_min = args.eps_min
        
        self.policy_type = args.policy
        
        self.device = torch.device("cpu")
        
        self.learn_step_counter = 0

        #self.policy = Netowrk(num_obs, num_action, args.hidden_size).to(device)
        #self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
        #self.target = QNetowrk(num_obs, num_action, args.hidden_size).to(device)
        #self.target_optim = Adam(self.target.parameters(), lr=args.lr)
        

    def select_action(self, obs, evaluate=False):
        '''
        obs = np.asarray(obs)
        observation = Variable(torch.from_numpy(obs)).float().to(self.device)
        
        a = self.policy.sample(observation)

        if episode <= OBSERVE:
          
        a_t = np.zeros([ACTIONS])
                
        return a
        '''
        if np.ramdom.random() < self.epsilon:
            state = torch.Tensor([observation]).to(device)
            advantage = self.policy.advantage(state)
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def update_parameters(self, memory, updates):

        obs_batch, action_batch, reward_batch, last_obs_batch, terminal_batch  = memory.sample(batch_size=self.batch_size)

        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        last_obs_batch = torch.FloatTensor(last_obs_batch).to(self.device)
        terminal_batch = torch.FloatTensor(terminal_batch).to(self.device)
        
        indices = np.arange(self.batch_size)

        q_pred = self.policy(obs_batch)[indices, action_batch]
        q_next = self.target(last_obs_batch)

        max_actions = torch.argmax(self.batch_size)

        q_target = reward_batch +self.gamma * q_next[indices, max_actions]

        q_next[dones] = 0.0
        self.policy.optim.zero_grad()

        loss = self.policy.crit(q_target, q_pred)
        loss.backward()

        self.policy.optim.step()

        self.epsilon = self.epsilon - self.epilon_decay if self.epsilon > self.eps_min else self.eps_min
        self.learn_step_counter += 1
