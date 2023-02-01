import os
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym.envs.registration import register
from ppo.env_PID_Curr import ArmEnv

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
    if (self.action_count % 300) == 0: 
      self.int1 = np.random.randint(0,2)
      self.int2 = np.random.randint(0,2)
    if self.int1 == 0 and self.int2 == 0: 
      j1 = np.random.uniform(0.3, 0.7)
      j2 = np.random.uniform(-0.1, -0.9)
    elif self.int1 == 0 and self.int2 == 1:
      j1 = np.random.uniform(-0.3, -0.7)
      j2 = np.random.uniform(-0.1, -0.9)
    elif self.int1 == 1 and self.int2 == 0:
      j1 = np.random.uniform(-0.1, -0.9)
      j2 = np.random.uniform(0.3, 0.7)
    else: 
      j1 = np.random.uniform(0.1, 0.9)
      j2 = np.random.uniform(0.3, 0.7)

    self.action_count +=1
    return np.array([j1, j2])


register(
    id="ArmGymJointControl-v1",
    entry_point=ArmGymJointControl,
    max_episode_steps=300,
)
