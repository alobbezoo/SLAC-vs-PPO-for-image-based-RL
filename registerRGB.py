# Gym enviroment registration
# Random action steps

import os
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym.envs.registration import register
from env import ArmEnv

from stable_baselines3.common.env_checker import check_env

class ArmGymJointControl(gym.Env):
  """Custom Environment that follows gym interface"""

  def __init__(self, ArmEnv=ArmEnv()):
    super(ArmGymJointControl, self).__init__()
    # They must be gym.spaces objects
    self.env = ArmEnv
    self.action_space = spaces.Box(low=-1, high=1, shape=(ArmEnv.action_dim,), dtype=np.float64)
    # self.observation_space = spaces.Box(low=-np.pi * 2, high=np.pi * 2, shape=(ArmEnv.state_dim,), dtype=np.float64)
    self.observation_space = spaces.Box(low=np.zeros((3,64,64)), high=255*np.ones((3,64,64)), shape=(3,64,64), dtype=np.uint8)
    self.action_count = 0
    self.int1 = 0
    self.int2 = 0
    self.ep_reward = 0


  def step(self, action):
    state, reward, done = self.env.step(action)
    state_image = self.stateRender() 
    info = {'is_success': np.array(float(done))}

    return state_image, reward, done , info

  def reset(self):
    state = self.env.reset()
    state_image = self.stateRender() 
    return state_image

  def render(self, mode='human', close=False):
    self.env.render()

  def stateRender(self, mode='human', close=False):
    image = self.env.render()
    return image
 
  def sample_action(self):
    return (2*np.random.rand(2)-1)    # two radians
    return np.array([j1, j2])

register(
    id="ArmGymJointControl-v1",
    entry_point=ArmGymJointControl,
    max_episode_steps=300,
)

# env = gym.make("ArmGymJointControl-v1")
# check_env(env, warn=True)
