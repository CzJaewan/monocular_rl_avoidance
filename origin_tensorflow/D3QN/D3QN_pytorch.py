from __future__ import print_function
from GazeboWorld import GazeboWorld
import os
import sys
import tensorflow as tf
import random
import numpy as np
import time
import rospy
import argparse
import torch
import torch.nn as nn
from pytorch_models.D3QN import D3QN
from pytorch_models.network import QNetwork, Network
from pytorch_models.momory import ReplayMemory

from collections import deque

parser = argparse.ArgumentParser(description='PyTorch D3QN Args')

parser.add_argument('--env_name', default="GazeboWorld",
                    help='Environment name (default: Stage)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(\tau) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
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
parser.add_argument('--cuda', type=bool, default=True,
                    help='run on CUDA (default: True)')
parser.add_argument('--epoch', type=int, default=1,
                    help='Epoch (default: 1)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')     
parser.add_argument('--policy_path',  default="multi_stage_v9", 
                    help='policy_path (default: multi_stage)')     # V6 : Reward design, expand robot radius, no terminal penalty      # v7 angular penalty, terminal penalty, expand robot radius         
parser.add_argument('--epsilon',  type=float, default=0.1, 
                    help='epsilon (default: 0.1)')  
parser.add_argument('--epsilon_decay',  type=float, default=1e-8, 
                    help='epsilon_decay (default: 1e-8)')  
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
BATCH = 4 # size of minibatch
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

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
	tf.summary.scalar('mean', mean)
	with tf.name_scope('stddev'):
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	tf.summary.scalar('stddev', stddev)
	tf.summary.scalar('max', tf.reduce_max(var))
	tf.summary.scalar('min', tf.reduce_min(var))
	tf.summary.histogram('histogram', var)

def trainNetwork():
	sess = tf.InteractiveSession()
	with tf.name_scope("OnlineNetwork"):
		online_net = QNetwork(sess)
	with tf.name_scope("TargetNetwork"):
		target_net = QNetwork(sess)
	rospy.sleep(1.)

	reward_var = tf.Variable(0., trainable=False)
	reward_epi = tf.summary.scalar('reward', reward_var)
	# define summary
	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter('./logs', sess.graph)

	# Initialize the World
	env = GazeboWorld()
	print('Environment initialized')

	# Initialize the buffer
	D = deque()

	# get the first state 
	depth_img_t1 = env.GetDepthImageObservation()
	depth_imgs_t1 = np.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), axis=2)
	terminal = False
	
	# saving and loading networks
	trainables = tf.trainable_variables()
	trainable_saver = tf.train.Saver(trainables)
	sess.run(tf.global_variables_initializer())

	checkpoint = tf.train.get_checkpoint_state("saved_networks/Q")
	
	print('checkpoint:', checkpoint)
	if checkpoint and checkpoint.model_checkpoint_path:
		trainable_saver.restore(sess, checkpoint.model_checkpoint_path)
		print("Successfully loaded:", checkpoint.model_checkpoint_path)
	else:
		print("Could not find old network weights")
		
	# start training
	episode = 0
	epsilon = INITIAL_EPSILON
	r_epi = 0.
	times = 0
	T = 0
	rate = rospy.Rate(5)
	print('Number of trainable variables:', len(trainables))
	targetOps = updateTargetGraph(trainables,TAU)
	loop_time = time.time()
	last_loop_time = loop_time
	total_reward = 0.
	
	while episode < MAX_EPISODE and not rospy.is_shutdown():
		env.ResetWorld()
		times = 0
		r_epi = 0.
		terminal = False
		reset = False
		loop_time_buf = []
		action_index = 0

		while not reset and not rospy.is_shutdown():
			depth_img_t1 = env.GetDepthImageObservation()
			depth_img_t1 = np.reshape(depth_img_t1, (DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1))
			depth_imgs_t1 = np.append(depth_img_t1, depth_imgs_t1[:, :, :(IMAGE_HIST - 1)], axis=2)
			reward_t, terminal, reset = env.GetRewardAndTerminate(times)
			if times > 0 :
				D.append((depth_imgs_t, a_t, reward_t, depth_imgs_t1, terminal))
				if len(D) > REPLAY_MEMORY:
					D.popleft()

			depth_imgs_t = depth_imgs_t1

			# choose an action epsilon greedily
			a = sess.run(online_net.readout, feed_dict = {online_net.state : [depth_imgs_t1]})
			readout_t = a[0]
			a_t = np.zeros([ACTIONS])
			if episode <= OBSERVE:
				action_index = random.randrange(ACTIONS)
				a_t[action_index] = 1
			else:
				if random.random() <= epsilon:
					print("----------Random Action----------")
					action_index = random.randrange(ACTIONS)
					a_t[action_index] = 1
				else:
					action_index = np.argmax(readout_t)
					a_t[action_index] = 1
			# Control the agent
			env.Control(action_index)

			if episode > OBSERVE :
				#print("D :", D)
				#print("BATCH :", BATCH)
				#print("D SIZE :", len(D))				

				# # sample a minibatch to train on
				minibatch = random.sample(D, BATCH)
				y_batch = []
				# get the batch variables
				depth_imgs_t_batch = [d[0] for d in minibatch]
				a_batch = [d[1] for d in minibatch]
				r_batch = [d[2] for d in minibatch]
				depth_imgs_t1_batch = [d[3] for d in minibatch]
				Q1 = online_net.readout.eval(feed_dict = {online_net.state : depth_imgs_t1_batch})
				Q2 = target_net.readout.eval(feed_dict = {target_net.state : depth_imgs_t1_batch})
				for i in range(0, len(minibatch)):
					terminal_batch = minibatch[i][4]
					# if terminal, only equals reward
					if terminal_batch:
						y_batch.append(r_batch[i])
					else:
						y_batch.append(r_batch[i] + GAMMA * Q2[i, np.argmax(Q1[i])])

				#Update the network with our target values.
				online_net.train_step.run(feed_dict={online_net.y : y_batch,
													online_net.a : a_batch,
													online_net.state : depth_imgs_t_batch })
				updateTarget(targetOps, sess) # Set the target network to be equal to the primary network.

			r_epi = r_epi + reward_t

			times = times + 1
			#print("t", times)
			T += 1
			last_loop_time = loop_time
			loop_time = time.time()
			loop_time_buf.append(loop_time - last_loop_time)
			rate.sleep()

			# scale down epsilon
			if epsilon > FINAL_EPSILON and episode > OBSERVE:
				epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
		total_reward = total_reward + r_epi
		
		#  write summaries
		if episode > OBSERVE:
			summary_str = sess.run(merged_summary, feed_dict={reward_var: r_epi})
			summary_writer.add_summary(summary_str, episode - OBSERVE)

		# save progress every 500 episodes
		if (episode+1) % 500 == 0 :
			trainable_saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = episode)

		if len(loop_time_buf) == 0:
			print("EPISODE", episode, "/ REWARD", r_epi, "/ steps ", T, "/ TOTAL_REWARD", total_reward/episode)
		else:
			print("EPISODE", episode, "/ REWARD", r_epi, "/ steps ", T, "/ TOTAL_REWARD", total_reward/episode,
				"/ LoopTime:", np.mean(loop_time_buf))

		episode = episode + 1	


def trainRun(env, agent, policy_path, args):
	save_interval = args.save_interval
	
	total_numsteps = 0
	updates = 0

    memory = ReplayMemory(args.replay_size, args.seed)

	avg_reward = 0

	for i_episode in range(args.num_steps):

        episode_reward = 0
        episode_steps = 0
        done = False   

		#reset env

		#get observation

		# epi start
		step = 0
		while not done and not rospy.is_shutdown():
			step += 1

			action = agent.select_action(state)

		if len(memory) > args.batch_size:
			# Number of updates per step in environment
			for i in range(args.updates_per_step):
				# Update parameters of all the networks
				if step % 10 == 0 and step > 10 and args.eval == False:
					#critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

				updates += 1
		
		#env.Control(action)
		rospy.sleep(0.001)

		#get reward
		episode_steps += 1
		total_numsteps += 1
		episode_reward += reward

		#get observation for next state

		if args.eval == False:
			#memory.push(obs, reward, action, n_obs, done) # Append transition to memory

        #state = next_state

       	if total_numsteps > args.num_steps:
            break        
            
        avg_cnt += 1

		if i_episode != 0 and i_episode % save_interval == 0 and args.eval == False:

			torch.save(agent.policy.state_dict(), policy_path + '/policy_epi_{}'.format(i_episode))
			print('[Save] policy model saved when update {} times'.format(i_episode))
			torch.save(agent.target.state_dict(), policy_path + '/critic_1_epi_{}'.format(i_episode))
			print('[Save] target model saved when update {} times'.format(i_episode))

        print("[RESULT] rank: {}, Episode: {}, episode steps: {}, reward: {}, result: {}".format(env.index, i_episode, episode_steps, round(reward, 2), result))

        avg_reward += round(episode_reward, 2)
		
        if avg_cnt % 100 == 0:
            print("[RESULT] Average reward: {}".format(avg_reward/avg_cnt))


	

if __name__ == "__main__":

	print("[Start] D3QN Trainning")
	print("[Ready] Environment is", args.env_name)
	env = GazeboWorld()
	print("[Start] Environment is", args.env_name)
	
	policy_path = args.policy_path
	print("[Setting] Policy Path :", args.policy_path)

	reward = None
	
	obs_size = [DEPTH_IMAGE_WIDTH, DEPTH_IMAGE_HEIGHT, IMAGE_HIST]
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

	try:
		trainRun(env=env, agent=agent, policy_path=policy_path, args=args)
	except KeyboardTnterrupt:
		pass

	#main()

