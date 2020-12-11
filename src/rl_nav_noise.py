#!/usr/bin/env python3

import numpy as np
import numpy.random as npr
import os
import tensorflow
#from tensorflow.initializers import random_uniform

class noiseOU(object):
	"""docstring for Ornstein Uhlenbeck noise"""
	def __init__(self, action_dimension, theta=0.2, sigma=0.15, mu=0):
		self.action_dimension = action_dimension
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.state = np.ones(self.action_dimension) * self.mu
		self.reset()
		
	def reset(self):
		self.state = np.ones(self.action_dimension) * self.mu	

	def noise(self):
		x = self.state
		dx = self.state * (self.mu - x) + self.sigma * nr.randn(len(x))
		self.state = x + dx
		return self.state

if __name__ == "__main__":
	ou = noiseOU(3)
	states = []
	for i in range(1000):
		states.append(ou.noise())
		
#	def __call__(self):
#		x = self.x_prev + self.theta *(self.mu-self.x_prev)*self.dt +\
#			self.sigma*np.sqrt(self.dt)*npr.normal(size=self.mu.shape)
			
#		self.x_prev = x
#		return x
