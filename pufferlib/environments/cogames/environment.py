from pdb import set_trace as T
import numpy as np
import functools

import gymnasium as gym

import pufferlib
import pufferlib.emulation
import pufferlib.environments
from pufferlib.pufferlib import set_buffers

from mettagrid.envs import MettaGridEnv

def env_creator(name='machina_1'):
    return functools.partial(make, name)

def make(name, render_mode='rgb_array', buf=None, seed=0):
    '''Atari creation function'''
    from cogames import game

    # Load a game configuration
    config = game.get_game(name)

    # Create environment
    env = PufferMettaGridEnv(env_cfg=config)
    set_buffers(env, buf)
    return env

class PufferMettaGridEnv(MettaGridEnv):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info = {k: v for k, v in pufferlib.unroll_nested_dict(info)
            if 'action' not in k}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info = {k: v for k, v in pufferlib.unroll_nested_dict(info)
            if 'action' not in k}
        info = {k: v for k, v in info.items() if 'action' not in k}
        return obs, reward, terminated, truncated, info

