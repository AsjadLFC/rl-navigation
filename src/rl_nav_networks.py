#!/usr/bin/env python

import os
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, BatchNormalization, Activation

class CriticNetwork(keras.Model):
	def __init__(self, state_dim, action_dim, name='critic'):
		super(CriticNetwork, self).__init__()
		self.dimensions1 = state_dim
		self.dimensions2 = 1080
		self.dimensions3 = 400
		self.dimensions4 = 128
		
		self.filepath = "/home/asohail2/ros/src/rl_navigation/src/models/"
		self.model_name = name
		self.checkpoint_file = os.path.join(self.filepath, self.model_name+'_ddpg_saved_model')
		
		self.full_con1 = Dense(self.dimensions1, activation='relu')
		self.full_con2 = Dense(self.dimensions2, activation='relu')
		self.full_con3 = Dense(self.dimensions3, activation='relu')
		self.full_con4 = Dense(self.dimensions4, activation='relu')
		
		self.r = Dense(1, activation=None, bias_initializer=initializers.RandomUniform(-0.003, 0.003),
										kernel_initializer=initializers.RandomUniform(-0.003, 0.003))
		self.a = Dense(1, activation=None, bias_initializer=initializers.RandomUniform(-0.003, 0.003),
										kernel_initializer=initializers.RandomUniform(-0.003, 0.003))
		
	def call(self, state, action):
		value_action = self.full_con1(tf.concat([state, action], axis=1))
		value_action = self.full_con4(value_action)
		
		r = self.r(value_action)
		
		return r
		
class ActorNetwork(keras.Model):
	def __init__(self, state_dim, action_dim, action_bound, name='actor'):
		super(ActorNetwork, self).__init__()
		self.dimensions1 = state_dim
		self.dimensions2 = 1080
		self.dimensions3 = 400
		self.dimensions4 = 128
		self.num_actions = action_bound
		
		self.filepath = "/home/asohail2/ros/src/rl_navigation/src/models/"
		self.model_name = name
		self.checkpoint_file = os.path.join(self.filepath, self.model_name+'_ddpg_saved_model')

		self.full_con1 = Dense(self.dimensions1)
		self.full_con1_bn = BatchNormalization(scale=True , center=True , epsilon=1e-5)
		self.full_con1 = Activation("relu")
		
		self.full_con2 = Dense(self.dimensions2)
		self.full_con2_bn = BatchNormalization(scale=True , center=True , epsilon=1e-5)
		self.full_con2 = Activation("relu")
		
		self.full_con3 = Dense(self.dimensions3)
		self.full_con3_bn = BatchNormalization(scale=True , center=True , epsilon=1e-5)
		self.full_con3 = Activation("relu")
		
		self.full_con4 = Dense(self.dimensions4)
		self.full_con4_bn = BatchNormalization(scale=True , center=True , epsilon=1e-5)
		self.full_con4 = Activation("relu")

		self.speed = Dense(1, activation="tanh", bias_initializer=initializers.RandomUniform(-0.003, 0.003),
						kernel_initializer=initializers.RandomUniform(-0.003, 0.003))
		self.angular = Dense(1, activation="tanh", bias_initializer=initializers.RandomUniform(-0.003, 0.003),
						kernel_initializer=initializers.RandomUniform(-0.003, 0.003))
		
	def call(self, state):
		prob = self.full_con1(state)
		prob = self.full_con2(prob)
		prob = self.full_con3(prob)
		prob = self.full_con4(prob)
		
		mu_speed = self.speed(prob)
		mu_angular = self.angular(prob)
		
		return tf.concat([mu_speed, mu_angular], axis=1)
