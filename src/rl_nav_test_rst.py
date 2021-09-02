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
		# ~ self.velocity_publish = rospy.Publisher('/GETjag/cmd_vel', Twist, queue_size=20)
		self.velocity_publish = rospy.Publisher('/cmd_vel', Twist, queue_size=20)
		# ~ self.scan_data = rospy.Subscriber('/base_scan', LaserScan, self.scan_message, queue_size=1)
		
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
		self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
		# ~ self.rate = rospy.Rate(40)
		
		self.position = Pose()
		
		self.action_space = spaces.Discrete(3)
		
		self.model_coord = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
				
		self._seed()
		
	def scan_message(self, msg):
		self.data = msg
		
	def calculate_observation(self):
		min_range = 0.45 # 0.72, 0.45
		scan_num = []
		done = False
		
		data = None
		while data is None:
			try:
				# ~ data = rospy.wait_for_message('/GETjag/laser_scan_front', LaserScan, timeout=5)
				self.check_connection()
				data = rospy.wait_for_message('/base_scan', LaserScan, timeout=10)
			except:
				# ~ print("not getting the scan data.. ")
				pass
		
		# ~ data = self.data
		
		for i in range(0, len(data.ranges)):
			if data.ranges[i] == float('Inf'):
				scan_num.append(3.5)
			elif np.isnan(data.ranges[i]):
				scan_num.append(0)
			else:
				scan_num.append(data.ranges[i])
		
		## [135 - 270] , [810 - 945]
		# ~ if sum(scan_num[155:250]) > sum(scan_num[830:925]):
			# ~ right_neg = True
				
		obstacle_min_range = round(min(scan_num), 2)
		obstacle_angle = np.argmin(scan_num)
		
		d_right = sum(scan_num[0:360])/360
		d_left = sum(scan_num[720:1080])/360
		d_right_front = sum(scan_num[360:540])/180
		d_left_front = sum(scan_num[540:720])/180
	#	d_front = sum(scan_num[520:560])/40
		
		if min_range > min(scan_num) > 0:
			done = True
				
		# ~ return scan_num + [obstacle_min_range, d_right, d_left], done, right_neg
	#	return scan_num + [obstacle_min_range, d_right, d_right , d_right , d_right,
	#						d_right, d_left, d_left, d_left, d_left, d_left,
	#						d_right_front, d_right_front, d_left_front, d_left_front, obstacle_angle], done
        return scan_num, done
			
	def step(self, action):

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

        angular_reward = 0
        if abs(ang_velocity) > 0:
            angular_reward = 35
		
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
		self.reset_cmd_vel()
		
		# take observation 
		data, done = self.calculate_observation()
		
		# pause simulation
		self.pauseSim()
			
		return data, done
	
	def pauseSim(self):
		# ~ return
		rospy.wait_for_service('/gazebo/pause_physics')
		try:
			self.pause()
		except (rospy.ServiceException) as e:
			rospy.loginfo("pause physics failed")
		
	def unpauseSim(self):
		# ~ return
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
		# ~ return
		self.reset_cmd_vel()
		
		rospy.wait_for_service('/gazebo/delete_model')
		rospy.wait_for_service('/gazebo/reset_world')
		
		try:
			self.reset_proxy()
		except (rospy.ServiceException) as e:
			rospy.loginfo("reset_simulation failed to execute")

