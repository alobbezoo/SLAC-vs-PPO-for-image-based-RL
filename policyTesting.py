"""
Script for testing pre-trained ppo model, change path accordingly

"""

# Imports
from stable_baselines3 import PPO
import registerRGB
import gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from ppo.optunaHyp import make_env

from slac.optunaHyp import SLAC

import os

# Load Trained Agent: PPO
homeDir = str(os.getcwd() + "/")
pathDir = str(homeDir + "ppo/2022-05-16/optimized/best_model")

vec_env = make_env(env_id="ArmGymJointControl-v1", n_envs=3, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=3)

model = PPO.load(pathDir)

# Evaluate the trained agent:
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
