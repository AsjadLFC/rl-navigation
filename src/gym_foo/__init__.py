#!/usr/bin/env python
from gym.envs.registration import register

register(
	id='rl-navigation-v0',
	entry_point='gym_foo.envs:rl_nav',
)
