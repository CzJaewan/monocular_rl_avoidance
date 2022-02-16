import torch
import logging

import os
import sys
import socket

from .net import QNetwork, Network

from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
import random

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.optim import Adam

#from pytorch_models.network import QNetwork, Network


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

        self.action_space = [0, 6]
        self.action_size = 7

        self.seed = random.seed(args.seed)

        self.policy = Network(num_obs, num_action, args.hidden_size, self.seed).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
        self.target = Network(num_obs, num_action, args.hidden_size, self.seed).to(self.device)
        self.target_optim = Adam(self.target.parameters(), lr=args.lr)
        

    def select_action(self, obs, evaluate=False):

        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        self.policy.eval()
        with torch.no_grad():
            Q, A, V = self.policy.forward(state)
        self.policy.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            #action = np.argmax(Q.cpu().data.numpy())
            action = torch.argmax(Q, 1)
            #print("V : ", V)
            #print("A : ", A)
            print("Q : ", Q)
            #print("Q TYPE : ", type(Q))
            #print("Q_NUM : ", Q.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
            print("random")

        return action.item()

    def update_parameters(self, memory):

        obs_batch, action_batch, reward_batch, next_obs_batch, terminal_batch  = memory.sample(batch_size=self.batch_size)

        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        terminal_batch = torch.LongTensor(terminal_batch).to(self.device)
    
        #batch_idx = torch.arange(self.batch_size, dtype=torch.long).to(self.device)
        #----------------------------------------------------------------------------------------
        target_q, target_a, target_v = self.target.forward(next_obs_batch)

        q, _, _ = self.policy.forward(obs_batch)

        q_n, _, _=self.policy.forward(next_obs_batch)

        pred_actions = torch.argmax(q_n, 1)
        pred_actions = pred_actions.reshape(-1,1)
        reward_batch = reward_batch.reshape(-1,1)
        terminal_batch = terminal_batch.reshape(-1,1)

        target = reward_batch + self.gamma*target_q.gather(1, pred_actions)*(1-terminal_batch)     
        action_batch = action_batch.reshape(-1,1)
        loss = F.mse_loss(target, q.gather(1, action_batch))
        
        #----------------------------------------------------------------------------------------
        
        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

        self.update_network_parameters()
        self.decrement_epsilon()

        print("[D3QN] Updata")


    def decrement_epsilon(self):
        
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.epilon_decay
        else: 
            self.epsilon = self.eps_min

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_params, policy_params in zip(self.target.parameters(), self.policy.parameters()):
            target_params.data.copy_(tau * policy_params + (1 - tau) * target_params)
