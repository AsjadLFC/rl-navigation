#!/usr/bin/env python

import gym
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

from tensorflow.keras.optimizers import Adam
from gym import spaces

from rl_nav_noise import noiseOU
from rl_nav_buffer import ReplayBuffer
from rl_nav_networks import CriticNetwork, ActorNetwork

REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 200
GAMMA = 0.99

class DDPG:
	"""docstring for DDPG"""
	def __init__(self, state_dim, action_dim, env, actor_lr=0.0001, critic_lr=0.0002):
		self.name = 'DDPG'
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.environment = env
		
		self.batch_size = BATCH_SIZE
		self.replay_start_size = REPLAY_START_SIZE
		self.gamma = 0.99
		self.tau = 0.001
		
		self.array = np.zeros((1083))
		self.action_space = 2
		self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE, self.array.shape, 2)
		self.noise = 1.0
		
		self.actor = ActorNetwork(self.state_dim, self.action_dim, self.action_space, name='actor')
		self.critic = CriticNetwork(self.state_dim, self.action_dim, name='critic')
		self.target_actor = ActorNetwork(self.state_dim, self.action_dim, self.action_space, name='target_actor')
		self.target_critic = CriticNetwork(self.state_dim, self.action_dim, name='target_critic')
		
		self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
		self.critic.compile(optimizer=Adam(learning_rate=critic_lr))
		self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr))
		self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr))
		
		self.update_network_parameters(tau=1)
		
	def update_network_parameters(self, tau=None):
		if tau is None:
			tau = self.tau
			
		weights = []
		targets = self.target_actor.weights
		for i, weight in enumerate(self.actor.weights):
			weights.append(weight * tau + targets[i]*(1-tau))
		self.target_actor.set_weights(weights)
			
		weights = []
		targets = self.target_critic.weights
		for i, weight in enumerate(self.critic.weights):
			weights.append(weight * tau + targets[i]*(1-tau))
		self.target_critic.set_weights(weights)
		
	def remember(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)

	def choose_action(self, observation, evaluate=False):
		state = tf.convert_to_tensor([observation], dtype=tf.float32)
		actions = self.actor(state)
#		print(f"actions before : {actions}")
		if not evaluate:
			actions += tf.random.normal(shape=[self.action_space], mean = 0.0, stddev = self.noise)
#		print("actions after: ", actions)
		
		actions = tf.clip_by_value(actions, -0.5, 0.5)
		
		return actions
		
	def learn(self):
		
		# 1. first we check if we have enough samples to learn
		if self.memory.mem_cntr < self.replay_start_size:
			return
			
		# 2. Then sample experience and convert them into tensors
		state, action, reward, new_state, dones = self.memory.sample_buffer(self.batch_size)
		
		states = tf.convert_to_tensor(state, dtype=tf.float32)
		states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
		rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
		actions = tf.convert_to_tensor(action, dtype=tf.float32)
		dones = tf.convert_to_tensor(dones, dtype=tf.bool)
		
		# 3. Bellman equation & calculating critic loss
		with tf.GradientTape() as tape:
			target_actions = self.target_actor(states_)
			critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
			critic_value = tf.squeeze(self.critic(states, actions), 1)
			target = reward + self.gamma*critic_value_
			critic_loss = keras.losses.MSE(target, critic_value)
			
		critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
		self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))
		
		with tf.GradientTape() as tape:
			new_policy_actions = self.actor(states)
			actor_loss = -self.critic(states, new_policy_actions)
			actor_loss = tf.math.reduce_mean(actor_loss)
			
		actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
		self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
		
		self.update_network_parameters()
			
#	def save_models(self):
#		print('... save model ...')
#		self.actor.save_weights(self.actor.checkpoint_file)
#		self.target_actor.save_weights(self.target_actor.checkpoint_file)
#		self.critic.save_weights(self.critic.checkpoint_file)
#		self.target_critic.save_weights(self.target_critic.checkpoint_file)
		
		
#	def load_models(self):
#		print('... load model ...')
#		self.actor.load_weights(self.actor.checkpoint_file)
#		self.target_actor.load_weights(self.target_actor.checkpoint_file)
#		self.critic.load_weights(self.critic.checkpoint_file)
#		self.target_critic.load_weights(self.target_critic.checkpoint_file)
