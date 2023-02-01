"""
Script for training PPO after ideal hyperparameters were found with optuna

# import ppo.registerRGB_PID_Curr
Import this enviroment (line 13) if testing with PID reward shaping and curriculum learning

"""

#Custom Imports
from ppo.optunaHyp import make_env
from ppo.customCnnShallow import CustomCNN
from ppo.optimize import SaveOnBestTrainingRewardCallback
from stable_baselines3.common import results_plotter
import registerRGB # ********


#Standard Imports
import gym
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack


# File Saving
from datetime import datetime
date = datetime.now().date()

homeDir = str(os.getcwd() + "/")

pathDir = homeDir + "ppo/" + str(date) + "/"
if not os.path.exists(pathDir):
    os.mkdir(pathDir)

logDir =  pathDir + "optimized/"
if not os.path.exists(logDir):
    os.mkdir(os.path.join(logDir))

imgDir = pathDir + "images/"
if not os.path.exists(imgDir):
    os.mkdir(imgDir)


N_ENVS = 3
ENV_ID = "ArmGymJointControl-v1"
DEVICE = "cpu" #"cuda"
EVAL_FREQ = 10e3
OPTIMIZED_N_TIMESTEPS = 10e6
VERBOSE = 2

net_arch =[{"pi": [256,256,256,256,256,256],
            "vf": [256,256,256,256,256,256],}]

kwargs = {
    "policy": "CnnPolicy",
    "device": DEVICE,
    "verbose": 2,
    "gamma": 0.9245385166136467,
    "gae_lambda": 0.8441517391672235,
    "learning_rate": 0.0008439681859532497,
    "ent_coef": 1e-06,
    "vf_coef": 0.75,
    "clip_range": 0.075,
    "max_grad_norm": 0.5,
    "batch_size": 1024,
    "n_steps": 16384,
    "n_epochs": 8,
    "policy_kwargs": {
        "net_arch": net_arch,
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs":dict(features_dim=1000),
    },
}

if __name__ == "__main__":


    vec_env = make_env(env_id=ENV_ID, n_envs=N_ENVS, seed=0, monitor_dir=logDir)
    vec_env = VecFrameStack(vec_env, n_stack=N_ENVS)

    print("kwargs: ", kwargs)
    print("n_envs: ", N_ENVS)

    model= PPO(env=vec_env, **kwargs)

    callback = SaveOnBestTrainingRewardCallback(check_freq=EVAL_FREQ, log_dir=logDir, verbose=VERBOSE)

    model.learn(total_timesteps=int(OPTIMIZED_N_TIMESTEPS), callback=callback)

    results_plotter.plot_results([logDir], OPTIMIZED_N_TIMESTEPS, results_plotter.X_TIMESTEPS, "PPO Reacher")
    plt.savefig(imgDir +'optimize_plot.png')
