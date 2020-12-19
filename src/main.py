#!/usr/bin/env python

import gym
import gym_foo
import numpy as np
import csv
import filter_env
import tensorflow as tf
import rospy
import gc

gc.enable()

from pathlib import Path
from datetime import datetime

from rl_nav_test_rst import rl_nav
from rl_nav_ddpg_main import DDPG

EPISODES = 10000
MAX_EP_STEPS = 500
state_dim = 1083
action_dim = 2

def main():	
#	environment = gym.make('rl-navigation-v0')
	rospy.init_node('ddpg_stages', anonymous=True)
	environment = rl_nav()
	agent = DDPG(state_dim, action_dim, environment)
	now = datetime.now()
	
	for episode in range(EPISODES):
			
		state, done = environment.reset()
		total_reward = 0
			
		print("--------------------episode number: ", episode)

		for step in range(MAX_EP_STEPS):
				
			action = agent.choose_action(state)[0]
			print(f"action: {action}")
			state_, reward, done = environment.step(action[1])
				
			total_reward += reward
			
			agent.remember(state, action, reward, state_, done)
			state = state_
				
			agent.learn()
			
			if done or step == MAX_EP_STEPS - 1:
				print("total reward: ", float(total_reward))
				with open(f'results/{now}results.csv', 'a', newline='') as file:
					file.write(f"{episode}, {total_reward}\n")
				break				

if __name__ == "__main__":
	main()
