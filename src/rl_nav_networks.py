#!/usr/bin/env python

import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, BatchNormalization, Activation

class CriticNetwork(keras.Model):
	def __init__(self, state_dim, action_dim, name='critic'):
		super(CriticNetwork, self).__init__()
		self.dimensions1 = state_dim
#		self.dimensions2 = action_dim
		self.dimensions2 = 1024
		self.dimensions3 = 512
		
		self.model_name = name
		
		self.full_con1 = Dense(self.dimensions1, activation='relu')
		self.full_con2 = Dense(self.dimensions2, activation='relu')
		self.full_con3 = Dense(self.dimensions3, activation='relu')
		
		self.r = Dense(1, activation=None)
		
	def call(self, state, action):
		value_action = self.full_con1(tf.concat([state, action], axis=1))
		value_action = self.full_con3(value_action)
		
		r = self.r(value_action)
		
		return r
		
class ActorNetwork(keras.Model):
	def __init__(self, state_dim, action_dim, action_bound, name='actor'):
		super(ActorNetwork, self).__init__()
		self.dimensions1 = state_dim
		self.dimensions2 = 1024
		self.dimensions3 = 512
		self.num_actions = action_bound
		
		self.model_name = name
		
		self.full_con1 = Dense(self.dimensions1)
		self.full_con1_bn = BatchNormalization(scale=True , center=True , epsilon=1e-5)
		self.full_con1 = Activation("relu")
		
		self.full_con2 = Dense(self.dimensions2)
		self.full_con2_bn = BatchNormalization(scale=True , center=True , epsilon=1e-5)
		self.full_con2 = Activation("relu")
		
		self.full_con3 = Dense(self.dimensions3)
		self.full_con3_bn = BatchNormalization(scale=True , center=True , epsilon=1e-5)
		self.full_con3 = Activation("relu")
		
		self.mu = Dense(self.num_actions, activation='tanh',
#		kernel_initializer=tf.random.uniform(-0.003, 0.003, dtype=tf.dtypes.float32),
#		bias_initializer=tf.random.uniform(-0.003, 0.003, dtype=tf.dtypes.float32)
		kernel_initializer='glorot_uniform',
		bias_initializer='zeros'
		)
		
	def call(self, state):
		prob = self.full_con1(state)
		prob = self.full_con2(prob)
		prob = self.full_con3(prob)
		
		mu = self.mu(prob)
		
		return mu
