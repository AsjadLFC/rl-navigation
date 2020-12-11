#!/usr/bin/env python

import gym
import rospy
import roslaunch
import numpy as np
import random
import time
import math

from gym import utils, spaces
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from gym.utils import seeding

from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState, GetModelStateRequest

class rl_nav(gym.Env):
	def __init__(self):
#		rospy.init_node('navigation_obstacles', anonymous=True)
		rospy.init_node('getLaserMsg', anonymous=True)
		self.velocity_publish = rospy.Publisher('/GETjag/cmd_vel', Twist, queue_size=5)
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		self.position = Pose()
		
#		self.model_coord = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
#		self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
		
#		self.screen_height = 640
#		self.screen_width = 1080
#		self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
		
		self._seed()
		
	def calculate_observation(self):
		min_range = 0.45
		scan_num = []
		done = False
		print("calculate observation: here")
		
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
		
		print("min range: ", obstacle_min_range)
		
		if min_range > min(scan_num) > 0:
			print("should be done.")
			done = True
				
		return scan_num + [obstacle_min_range, obstacle_angle], done
			
	def step(self, action):
		 
		max_angular_speed = 0.3
		ang_velocity = (action - 10) * max_angular_speed * 0.1
		
		velocity_command = Twist()
		velocity_command.linear.x = 0.2
#		velocity_command.linear.z = ang_velocity
		
		self.unpauseSim()

		self.velocity_publish.publish(velocity_command)
		time.sleep(0.1)

		state, done = self.calculate_observation()
		
		self.pauseSim()
		print("stop in step")
		
		angular_reward = (1 - abs(ang_velocity)) * 35
		
		if done:
			reward = -200
		else:
			reward = 70
		
		cumulative_reward = reward + angular_reward
		
		print("cumulative reward: ", cumulative_reward)
		
		return state, cumulative_reward, done
		
	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		rospy.wait_for_service('/gazebo/delete_model')
		rospy.wait_for_service('gazebo/reset_world')
		
		rospy.wait_for_service('/gazebo/reset_simulation')
		try:
			self.reset_proxy()
		except (rospy.ServiceException) as e:
			print("reset_simulation failed to execute")
		
		self.unpauseSim()
		
		data = self.calculate_observation()
		
		self.pauseSim()
			
		return data
	
	def pauseSim(self):
		rospy.wait_for_service('/gazebo/pause_physics')
		try:
			self.pause()
		except (rospy.ServiceException) as e:
			print("pause physics failed")
		
	def unpauseSim(self):
		rospy.wait_for_service('/gazebo/unpause_physics')
		try:
			self.unpause()
		except (rospy.ServiceException) as e:
			print("unpause physics failed")
