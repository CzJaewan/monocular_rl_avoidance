import torch
import numpy as np
from torch import optim
import torch.nn as nn
import random

def trans_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    #state = torch.from_numpy(state)
    #state = state.unsqueeze(0)

    return state

class ReplayBuffe():
    def __init__(self, max_size, input_shape, seed):
        random.seed(seed)
        self.mem_size = max_size
        self.mem_cntr = 0

        self.obs_memory = np.zero((self.mem_size, *input_shape), dtype = np.float32)
        self.new_obs_memory = np.zero((self.mem_size, *input_shape), dtype = np.float32)
        self.action_memory = np.zero((self.mem_size, *input_shape), dtype = np.int64)
        self.reward_memory = np.zero((self.mem_size, *input_shape), dtype = np.float32)
        self.terminal_memory = np.zero((self.mem_size, *input_shape), dtype = np.bool)

    def store_transition(self, obs, action, reward, next_obs, done):
        idx = self.mem_cntr % self.mem_size
        
        self.obs_memory[idx] = obs
        self.new_obs_memory[idx] = next_obs
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace = False)

        obs = self.obs_memory[batch]
        new_obs = self.new_obs_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return obs, new_obs, action, reward, terminal

class ReplayMemory():
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, frame, action, reward, n_frame, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (frame, action, reward, n_frame, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        frame, action, reward, n_frame, done = map(np.stack, zip(*batch))
        return frame, action, reward, n_frame, done

    def __len__(self):
        return len(self.buffer)
