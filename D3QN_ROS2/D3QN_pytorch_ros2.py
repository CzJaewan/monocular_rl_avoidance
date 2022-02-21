#from __future__ import print_function
from GazeboWorld_ros2 import GazeboWorld

import os
import sys
import socket

import random
import numpy as np
import time
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import threading

import argparse

import torch
import torch.nn as nn
from pytorch_models.memory import ReplayMemory
import pytorch_models.memory as Mem
from pytorch_models.D3QN import D3QN
from pytorch_models.net import QNetwork, Network
from torch.optim import Adam

from collections import deque

parser = argparse.ArgumentParser(description='PyTorch D3QN Args')

parser.add_argument('--env_name', default="GazeboWorld",
                    help='Environment name (default: Stage)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='target smoothing coefficient(\tau) (default: 0.001)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size (default: 4)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=300000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', type=bool, default=False,
                    help='run on CUDA (default: True)')
parser.add_argument('--epoch', type=int, default=1,
                    help='Epoch (default: 1)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')     
parser.add_argument('--policy_path',  default="multi_stage_v9", 
                    help='policy_path (default: multi_stage)')     # V6 : Reward design, expand robot radius, no terminal penalty      # v7 angular penalty, terminal penalty, expand robot radius         
parser.add_argument('--epsilon',  type=float, default=0.1, 
                    help='epsilon (default: 0.1)')  
parser.add_argument('--epsilon_decay',  type=float, default=0.000004995, 
                    help='epsilon_decay (default: 0.000004995)')  
parser.add_argument('--eps_min',  type=float, default=0.0001, 
                    help='epsilon min (default: 0.0001)')  
parser.add_argument('--save_interval',  type=int, default=300, 
                    help='policy save interval (default: 300)')  

args = parser.parse_args()

GAME = 'GazeboWorld'
ACTIONS = 7 # number of valid actions
SPEED = 2 # DoF of speed
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10. # timesteps to observe before training
EXPLORE = 20000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 10000 # number of previous transitions to remember
BATCH = 12 # size of minibatch
MAX_EPISODE = 20000
MAX_T = 200
DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
RGB_IMAGE_HEIGHT = 228
RGB_IMAGE_WIDTH = 304
CHANNEL = 3
TAU = 0.001 # Rate to update target network toward primary network
H_SIZE = 8*10*64
IMAGE_HIST = 4


def trainRun(env, agent, policy_path, args):
	save_interval = args.save_interval
	print("epsilon_decay : ",args.epsilon_decay)
	total_numsteps = 0
	updates = 0

	#memory set
	memory = ReplayMemory(args.replay_size, args.seed)

	avg_reward = 0
	avg_cnt = 0

	#rate = env.rate()

	loop_time = time.time()
	last_loop_time = loop_time

	for i_episode in range(args.num_steps):

		episode_reward = 0
		episode_steps = 0
		done = False
		reset = False
		loop_time_buf = []

		action_index = 0
		
		#reset env
		env.ResetWorld()
		
		#get observation
		depth_img_t1 = env.GetDepthImageObservation()
		state = np.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), axis=0)
		
		while not reset: #and not rclpy.is_shutdown():

			action = agent.select_action(state)

			print("action_index : ", action)
			env.Control(action)
			
			#get reward
			reward, done, reset, result = env.GetRewardAndTerminate(episode_steps)
			
			episode_steps += 1
			total_numsteps += 1
			episode_reward += reward

			#get observation for next state
			next_depth_img_t1 = env.GetDepthImageObservation()
			next_depth_img_t1 = np.reshape(next_depth_img_t1, (1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH))
			next_state = np.append(next_depth_img_t1, state[:(IMAGE_HIST - 1), :, :], axis=0)

			if args.eval == False:
				memory.push(state, action,reward, next_state, done) # Append transition to memory
				if(len(memory) == REPLAY_MEMORY):
					print('[Saved] memory full charge : buff size {}'.format(len(memory)))

			if len(memory) > args.batch_size:

				agent.update_parameters(memory)
			

			state = next_state
			last_loop_time = loop_time
			loop_time = time.time()
			loop_time_buf.append(loop_time - last_loop_time)
			time.sleep(0.1)

		if total_numsteps > args.num_steps:
			break        

		avg_cnt += 1

		if i_episode != 0 and i_episode % save_interval == 0 and args.eval == False:

			torch.save(agent.policy.state_dict(), policy_path + '/policy_epi_{}'.format(i_episode))
			print('[Save] policy model saved when update {} times'.format(i_episode))
			torch.save(agent.target.state_dict(), policy_path + '/critic_1_epi_{}'.format(i_episode))
			print('[Save] target model saved when update {} times'.format(i_episode))

		print("[RESULT] Episode: {}, episode steps: {}, reward: {}, result: {}".format(i_episode, episode_steps, round(episode_reward, 2), result))

		avg_reward += round(episode_reward, 2)
		
		if avg_cnt % 100 == 0:
			print("[RESULT] Average reward: {}".format(avg_reward/avg_cnt))

if __name__ == "__main__":
	print("[Start] D3QN Trainning")
	policy_path = args.policy_path
	print("[Setting] Policy Path :", args.policy_path)

	reward = None
	
	obs_size = IMAGE_HIST
	agent = D3QN(num_obs=obs_size, num_action=ACTIONS, args=args)
	
	if not os.path.exists(policy_path):
		os.makedirs(policy_path)
		print("[Create] Policy path :", args.policy_path)

	file_policy = policy_path + '/policy_epi_1'
	file_target = policy_path + '/target_epi_1'

	if os.path.exists(file_policy):
		print('[Load] Policy Model')

		state_dict = torch.load(file_policy)
		agent.policy.load_state_dict(state_dict)
	else:
		print('[Create] Policy Model')

	if os.path.exists(file_target):
		print('[Load] Target Model')

		state_dict = torch.load(file_target)
		agent.target.load_state_dict(state_dict)
	else:
		print('[Create] Target Model')

	print("[Ready] Environment is", args.env_name)
	env = GazeboWorld()
	print("[Start] Environment is", args.env_name)

	try:
		trainRun(env=env, agent=agent, policy_path=policy_path, args=args)
	except KeyboardInterrupt:
		pass

	#main()

