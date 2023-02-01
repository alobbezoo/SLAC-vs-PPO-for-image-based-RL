#Grasping Task Optimization: 

#Custom Imports
from slac.optunaHyp import SLAC

#Standard Imports
import gym
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

# VIDEO RENDERING: 
import base64
from pathlib import Path
from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'

# File Saving 
from datetime import datetime
date = datetime.now().date() 

homeDir = str(os.getcwd() + "/")

pathDir = homeDir + "slac/" + str(date) + "/"
if not os.path.exists(pathDir):
    os.mkdir(pathDir)

imgDir = pathDir + "images/"
if not os.path.exists(imgDir):
    os.mkdir(imgDir)

logDir =  pathDir + "optimized/"
if not os.path.exists(logDir):
    os.mkdir(os.path.join(logDir))

modelDir =  pathDir + "savedModels/"
if not os.path.exists(modelDir):
    os.mkdir(os.path.join(modelDir))


class TrainOpt():
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(
        self, 
        trial, 
        n_latent_timesteps= 3 * 10 ** 5,
        n_learning_timesteps= 2 * 10 ** 6,
        initial_collection_steps = 10 ** 4,
        eval_interval= 10 ** 4,
        env_id = "ArmGymJointControl-v1", 
        actionRepeat = 4, 
        pathDir = pathDir, 
        imgDir=imgDir, 
        logDir=logDir, 

    ):
        self.trial = trial
        self.n_latent_timesteps = n_latent_timesteps
        self.n_learning_timesteps = n_learning_timesteps
        self.initial_collection_steps = initial_collection_steps
        self.eval_interval = eval_interval
        self.env_id = env_id 
        self.actionRepeat = actionRepeat 

        self.pathDir = pathDir
        self.logDir = logDir
        self.imgDir = imgDir

        self.default_hyperparams = {
            "n_latent_timesteps": self.n_latent_timesteps, 
            "n_learning_timesteps": self.n_learning_timesteps,
            "eval_interval": self.eval_interval,
            "initial_collection_steps": initial_collection_steps
        }



    def optimal(self):
        gamma = self.trial.user_attrs['gamma']
        tau = self.trial.user_attrs['tau']
        lr_sac = self.trial.user_attrs['lr_sac']
        lr_latent = self.trial.user_attrs['lr_latent']
        batch_size_sac = self.trial.user_attrs['batch_size_sac']
        batch_size_latent = self.trial.user_attrs['batch_size_latent']
        hidden_units = self.trial.user_attrs['hidden_units']


        kwargs = {
            "gamma": gamma, 
            "tau": tau,
            "lr_sac": lr_sac, 
            "lr_latent": lr_latent,
            "batch_size_sac": batch_size_sac,
            "batch_size_latent": batch_size_latent,
            "hidden_units": hidden_units,
        }

        return kwargs

    def train(self):

        kwargs = self.default_hyperparams.copy()

        # Sample hyperparameters
        kwargs.update(self.optimal())
        print("\n \nKwargs are: ")
        print(kwargs)

        self.model= SLAC(env_id=self.env_id, **kwargs)

        self.model.learn()
        
        # results_plotter.plot_results([self.logDir], self.timeSteps, results_plotter.X_TIMESTEPS, "SLAC-Reacher")
        # plt.savefig(str(self.imgDir) +'/optimize_plot.png')
    

