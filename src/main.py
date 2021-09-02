#!/usr/bin/env python

import gym
import mlflow
import numpy as np
import csv
import filter_env
import tensorflow as tf
import rospy
import gc
import sys

gc.enable()

from pathlib import Path
from datetime import datetime

from rl_nav_test_rst import rl_nav
from rl_nav_ddpg_main import DDPG

EPISODES = 10000 #5000
MAX_EP_STEPS = 700
state_dim = 1081 # (1085,1), (53,30), (274,4)
action_dim = 2

def main():	
#	environment = gym.make('rl-navigation-v0')
	rospy.init_node('ddpg_stages', anonymous=False)
	environment = rl_nav()
	agent = DDPG(state_dim, action_dim, environment)
	now = datetime.now()
	is_training = False
	
	if is_training:
		for episode in range(EPISODES):
			
			state, done = environment.reset()
			total_reward = 0

			for step in range(MAX_EP_STEPS):
				
				action = agent.choose_action(state)
				# ~ print("action: ", action)
				state_, reward, done = environment.step(action)
				total_reward += reward
			
				agent.remember(state, action, reward, state_, done)
				state = state_
				agent.learn()
			
				if done or step == MAX_EP_STEPS - 1:
					print("--------------------episode number: ", episode)
					print("total reward: ", float(total_reward))
					with open(f'results/{now}results.csv', 'a', newline='') as file:
						file.write(f"{episode}, {total_reward}\n")
					break
		
			if episode == EPISODES-1:
				agent.save_models()
				
	else:
		print("Testing...")
		agent.load_models()
		print("Loaded...")
		state, done = environment.reset()
		
		for episode in range(2000):
			action = agent.choose_action(state)
			state_, reward, done = environment.step(action)
			state = state_
		
if __name__ == "__main__":
	main()
