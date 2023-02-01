#Custom Imports
from slac.timer import Timer
from slac.trainer import Trainer

# Standard Imports

import os 
import optuna
import plotly

import gym


from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances


# Custom Callback:
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import matplotlib.pyplot as plt
import numpy as np


import random

from typing import Any, Callable, Dict, Optional, Type, Union


# Custom Callback:
import matplotlib.pyplot as plt
import numpy as np

import random
import tensorflow as tf
import gym.spaces
import os
from typing import Any, Callable, Dict, Optional, Type, Union

from datetime import datetime
import torch

from slac.algo import SlacAlgorithm


date = datetime.now().date() 

homeDir = str(os.getcwd() + "/")

pathDir = homeDir + "slac/" + str(date) + "/"
if not os.path.exists(pathDir):
    os.mkdir(pathDir)

imgDir = pathDir + "images/"
if not os.path.exists(imgDir):
	os.mkdir(imgDir)

logDir =  pathDir + "optuna/"
if not os.path.exists(logDir):
	os.mkdir(logDir)

    
class SLAC(): 
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """
    def __init__(
        self, 
        # trainer hyperparameters
        n_latent_timesteps= 3 * 10 ** 4, # NOTE: Reduce for testing
        n_learning_timesteps= 2 * 10 ** 5,
        initial_collection_steps = 10 ** 3,
        eval_interval= 10 ** 4,

        logDir = logDir, 
        imgDir = imgDir,
        seed = 0,
        env_id = "ArmGymJointControl-v1", 
        actionRepeat = 4, 

        #algo hyperparameters
        gamma=0.99,
        tau=5e-3,
        lr_sac=3e-4,
        lr_latent=1e-4,
        batch_size_sac=256,
        batch_size_latent=32,
        hidden_units=(256, 256)
    ):

        self.logDir = logDir
        self.imgDir = imgDir
        self.env_id = env_id

        self.n_latent_timesteps = n_latent_timesteps
        self.n_learning_timesteps = n_learning_timesteps
        self.initial_collection_steps = initial_collection_steps
        self.eval_interval= eval_interval

        self.actionRepeat = actionRepeat
        self.seed = seed

        self.gamma = gamma
        self.tau=tau
        self.lr_sac=lr_sac
        self.lr_latent=lr_latent
        self.batch_size_sac=batch_size_sac
        self.batch_size_latent=batch_size_latent
        self.hidden_units=hidden_units

        self.env = gym.make(self.env_id)
        self.env_test = gym.make(self.env_id)

        setattr(self.env, 'action_repeat', self.actionRepeat)
        setattr(self.env_test, 'action_repeat', self.actionRepeat)


    def learn(self):
        self.algo = SlacAlgorithm(
            state_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            action_repeat=self.actionRepeat,
            device=torch.device("cuda"),
            seed=self.seed,

            gamma = self.gamma,
            tau = self.tau,
            lr_sac = self.lr_sac,
            lr_latent = self.lr_latent,
            batch_size_sac=self.batch_size_sac,
            batch_size_latent=self.batch_size_latent,
            hidden_units=self.hidden_units,
        )

        trainer = Trainer(
            env=self.env,
            env_test=self.env_test,
            algo=self.algo,
            log_dir=self.logDir,
            seed=self.seed,
            
            num_eval_episodes=5,
            num_sequences=8,

            eval_interval= self.eval_interval,
            num_steps=self.n_learning_timesteps, # task policy
            initial_learning_steps=self.n_latent_timesteps, # latent representation
            initial_collection_steps= self.initial_collection_steps, 
        )

        trainer.train()

        return trainer.optuna_return()

class OptunaFunc():
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """
    def __init__(
        self, 
        n_latent_timesteps = 10 ** 4, 
        n_learning_timesteps = 10 ** 5, 
        n_init_collection_steps = 5 * 10 ** 3,
        eval_interval =10 ** 3, 
        env_id = "ArmGymJointControl-v1",  

        logDir = logDir, 
        imgDir = imgDir
    ):

        self.logDir = logDir
        self.imgDir = imgDir
        self.env_id = env_id

        self.n_latent_timesteps = n_latent_timesteps
        self.n_learning_timesteps = n_learning_timesteps
        self.n_init_collection_steps = n_init_collection_steps
        self.eval_interval = eval_interval

        self.default_hyperparams = {
            "n_latent_timesteps": self.n_latent_timesteps, 
            "n_learning_timesteps": self.n_learning_timesteps,
            "eval_interval": self.eval_interval,
            "initial_collection_steps": n_init_collection_steps
}



    def sample_ppo_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sampler for A2C hyperparameters."""
        gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
        tau = trial.suggest_float("tau", 1e-3, 1e-2, log=True)
        lr_sac = trial.suggest_float("lr_sac", 1e-6, 1e-3, log=True)
        lr_latent = trial.suggest_float("lr_latent", 1e-6, 1e-3, log=True)
        batch_size_sac = 2**trial.suggest_int("batch_size_sac", 5, 9, log=True)
        batch_size_latent = 2**trial.suggest_int("batch_size_latent", 5, 9, log=True)
        
        net_arch_width = 2 ** trial.suggest_int("net_arch_width_int", 5, 9)
        net_arch_depth = trial.suggest_int("net_arch_depth", 2, 5)

        net_arch_array = np.ones(net_arch_depth,dtype=object)
        net = net_arch_array*net_arch_width
        hidden_units = tuple(net)
        # print(" \n hidden_units: ", hidden_units, "\n")

        # Display true values not shown otherwise
        trial.set_user_attr("gamma", gamma)
        trial.set_user_attr("tau", tau)
        trial.set_user_attr("lr_sac", lr_sac)
        trial.set_user_attr("lr_latent", lr_latent)
        trial.set_user_attr("batch_size_sac", batch_size_sac)
        trial.set_user_attr("batch_size_latent", batch_size_latent)
        trial.set_user_attr("hidden_units", hidden_units)


        return {
            "gamma": gamma,
            "tau": tau,
            "lr_sac": lr_sac,
            "lr_latent": lr_latent,
            "batch_size_sac": batch_size_sac,
            "batch_size_latent": batch_size_latent,
            "hidden_units": hidden_units,
        }

    def objective(self, trial: optuna.Trial) -> float:

        objective_trial = Timer()
        objective_trial.start()

        kwargs = self.default_hyperparams.copy()

        # Sample hyperparameters
        kwargs.update(self.sample_ppo_params(trial))

        print("\n \nKwargs are: ")
        print(kwargs)

        model = SLAC(env_id= self.env_id, **kwargs)

        nan_encountered = False

        try:
            return model.learn()

        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN
            print(e)
            nan_encountered = True

        # Tell the optimizer that the trial failed
        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        print("TRIAL Finished: Objective Trial Time")
        objective_trial.stop()

        return eval_callback.last_mean_reward



def Plotter(study):
    fig1 = plot_parallel_coordinate(study)
    fig2 = plot_optimization_history(study)
    fig3 = plot_param_importances(study)

    fig1.write_image(imgDir +"/plot_parallel_coordinate.jpeg")
    fig2.write_image(imgDir +"/plot_optimization_history.jpeg")
    fig3.write_image(imgDir +"/plot_param_importances.jpeg")


 
