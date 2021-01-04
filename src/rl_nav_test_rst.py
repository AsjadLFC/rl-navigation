#!/usr/bin/env python

import gym
import rospy
import roslaunch
import numpy as np
import random
import time
import math
import tensorflow as tf

from gym_foo.envs import gazebo_env
from gym import utils, spaces
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from gym.utils import seeding

from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState, GetModelStateRequest, SetModelState

class rl_nav():
	def __init__(self):
		self.velocity_publish = rospy.Publisher('/GETjag/cmd_vel', Twist, queue_size=20)
		
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
		self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
		
		self.position = Pose()
		
		self.action_space = spaces.Discrete(3)
		
		self.model_coord = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
#		self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
		
#		self.screen_height = 640
#		self.screen_width = 1080
#		self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
		
		self._seed()
		
	def calculate_observation(self):
		min_range = 0.73
		scan_num = []
		done = False
		
		data = None
		while data is None:
			try:
				data = rospy.wait_for_message('/GETjag/laser_scan_front', LaserScan, timeout=5)
			except:
				pass
		
		for i in range(len(data.ranges)):
			if data.ranges[i] == float('Inf'):
				scan_num.append(3.5)
			elif np.isnan(data.ranges[i]):
				scan_num.append(0)
			else:
				scan_num.append(data.ranges[i])
				
		obstacle_min_range = round(min(scan_num), 2)
		obstacle_angle = np.argmin(scan_num)
		
		if min_range > min(scan_num) > 0:
			done = True
				
		return scan_num + [obstacle_min_range, obstacle_angle], done
			
	def step(self, action):
#		max_angular_speed = 0.5
#		ang_velocity = round(float((action - 10) * max_angular_speed * 0.1), 4)

		ang_velocity = round(float(action[1]), 4)
		velocity_command = Twist()
		velocity_command.linear.x = action[0]
		velocity_command.angular.z = ang_velocity

		self.unpauseSim()
		
		self.velocity_publish.publish(velocity_command)
		rospy.sleep(0.1)
		
#		self.reset_cmd_vel()
		state, done = self.calculate_observation()
		
		self.pauseSim()

		angular_reward = abs(ang_velocity) * 10
		
		if done:
			reward = -500
		else:
			reward = round(float(action[0]), 2) * 70
		
		cumulative_reward = reward - angular_reward
		
		return state, cumulative_reward, done
		
	def _seed(self, seed=1):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		
		# reset
		self.resetSim()
		
		# unpause simulation
		self.unpauseSim()
		
		#reset to initial
#		self.check_connection()
#		self.reset_cmd_vel()
		
		# take observation 
		data = self.calculate_observation()
		
		# pause simulation
		self.pauseSim()
			
		return data
	
	def pauseSim(self):
		rospy.wait_for_service('/gazebo/pause_physics')
		try:
			self.pause()
		except (rospy.ServiceException) as e:
			rospy.loginfo("pause physics failed")
		
	def unpauseSim(self):
		rospy.wait_for_service('/gazebo/unpause_physics')
		try:
			self.unpause()
		except (rospy.ServiceException) as e:
			rospy.loginfo("unpause physics failed")
			
	def check_connection(self):
		rate = rospy.Rate(10)
		while(self.velocity_publish.get_num_connections() == 0):
			rospy.loginfo("no subscribers so we wait")
			rate.sleep();
		rospy.loginfo("cmd_pub connected")

	def reset_cmd_vel(self):
		cmd_vel = Twist()
		cmd_vel.linear.x = 0.0
		cmd_vel.angular.z = 0.0
		self.velocity_publish.publish(cmd_vel)
		
	def resetSim(self):
		rospy.wait_for_service('/gazebo/delete_model')
		rospy.wait_for_service('/gazebo/reset_world')
		try:
			self.reset_proxy()
		except (rospy.ServiceException) as e:
			rospy.loginfo("reset_simulation failed to execute")

