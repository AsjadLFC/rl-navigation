#!/usr/bin/env python

import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
	def __init__(self, state_dim, action_dim, name='critic'):
		super(CriticNetwork, self).__init__()
		self.dimensions1 = state_dim
		self.dimensions2 = action_dim
		
		self.model_name = name
		
		self.fc1 = Dense(self.dimensions1, activation='relu')
		self.fc2 = Dense(self.dimensions2, activation='relu')
		self.q = Dense(1, activation=None)
		
	def call(self, state, action):
		action_value = self.fc1(tf.concat([state, action], axis=1))
		action_value = self.fc2(action_value)
		
		q = self.q(action_value)
		
		return q
		
class ActorNetwork(keras.Model):
	def __init__(self, state_dim, action_dim, action_bound, name='actor'):
		super(ActorNetwork, self).__init__()
		self.dimensions1 = state_dim
		self.dimensions2 = action_dim
		self.num_actions = action_bound
		
		self.model_name = name
		
		self.fc1 = Dense(self.dimensions1, activation='relu')
		self.fc2 = Dense(self.dimensions2, activation='relu')
		self.mu = Dense(self.num_actions, activation='tanh')
		
	def call(self, state):
		prob = self.fc1(state)
		prob = self.fc2(prob)
		
		mu = self.mu(prob)
		
		return mu
