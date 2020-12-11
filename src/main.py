#!/usr/bin/env python

import gym
import gym_foo
import numpy as np
import filter_env
import tensorflow as tf
import rospy
import time
import gc
gc.enable()

from rl_nav_ddpg_main import DDPG

EPISODES = 1
MAX_EP_STEPS = 6000
state_dim = 12
action_dim = 1

def main():	
	environment = gym.make('rl-navigation-v0')
	agent = DDPG(state_dim, action_dim, environment)
	print('TESTING...', environment)
	
	for episode in range(EPISODES):
		state, done = environment.reset()
		total_reward = 0
		print("episode number: ", episode)

		for step in range(MAX_EP_STEPS):
			action = agent.choose_action(state)[0]
			state_, reward, done = environment.step(action)
			break
			total_reward += reward
			agent.remember(state, action, reward, state_, done)
			
			if done or step == MAX_EP_STEPS - 1:
				print("total reward: ", total_reward)
				break

		
#		for step in range(MAX_EP_STEPS):
#			age = agent.noise_action(state)[0]
#			print ("age: ", age)
#			state_, reward, done = environment.step(age)
#			print("number: ", step)
#			time_step = agent.perceive(state, age, reward, state_, done)
#			state = state_
#			total_reward += reward
			
#			if done or step == MAX_EP_STEPS - 1:
#				print("total reward: ", round(total_reward, 2))
#				break
			

if __name__ == "__main__":
	main()
