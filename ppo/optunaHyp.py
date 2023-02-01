#Custom Imports
from ppo.customCnnShallow import CustomCNN
from ppo.timer import Timer

# Standard Imports

import os 
import optuna
import plotly

import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import matplotlib.pyplot as plt
import numpy as np


from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

import random
import tensorflow as tf
import gym.spaces

from typing import Any, Callable, Dict, Optional, Type, Union

from datetime import datetime
import torch

date = datetime.now().date() 

homeDir = str(os.getcwd() + "/")

pathDir = homeDir + "ppo/" + str(date) + "/"
if not os.path.exists(pathDir):
    os.mkdir(pathDir)

imgDir = pathDir + "images/"
if not os.path.exists(imgDir):
	os.mkdir(imgDir)

logDir =  pathDir + "optuna/"
if not os.path.exists(logDir):
	os.mkdir(logDir)


def make_env(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[DummyVecEnv, SubprocVecEnv]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.
    :param env_id: the environment ID or the environment class
        n_envs = trial.suggest_int("n_envs", 3, 6, log=True) 
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
    )


class OptunaFunc():
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """
    def __init__(
        self, 
        n_timesteps = 1000, eval_freq =100, 
        n_eval_episodes = 3, env_id = "PandaReachDepth-v1", render = False, verbose = 2, 
        policy = "CnnPolicy", device="cuda", exponent_n_steps_min:int = 8, exponent_n_steps_max:int = 12,
        learning_rate_min:float = 0.0001, learning_rate_max:float = 0.001, ent_coef_min:float = 0.0000001, ent_coef_max = 0.1,
        logDir = logDir, imgDir = imgDir
        # policy_kwargs = {"features_extractor_class": CustomCNN,"features_extractor_kwargs":dict(features_dim=128),}
    ):

        self.n_timesteps = n_timesteps
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.env_id = env_id
        self.policy = policy
        self.device = device
        self.verbose = verbose
        self.exponent_n_steps_min = exponent_n_steps_min 
        self.exponent_n_steps_max = exponent_n_steps_max
        self.learning_rate_min = learning_rate_min
        self.learning_rate_max = learning_rate_max
        self.ent_coef_min = ent_coef_min
        self.ent_coef_max = ent_coef_max
        # self.policy_kwargs = policy_kwargs
        self.render = render
        
        self.logDir = logDir
        self.imgDir = imgDir

        self.default_hyperparams = {
            "policy": self.policy, 
            "device": self.device,
            "verbose": self.verbose,
}

    def sample_ppo_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sampler for A2C hyperparameters."""
        gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99, log=True)
        learning_rate = trial.suggest_float("lr", self.learning_rate_min, self.learning_rate_max, log=True)
        n_epochs = trial.suggest_int("n_epochs", 7, 13, log=True) 

        vf_coef = 0.75 #trial.suggest_float("vf_coef", 0.35, 1, log=True)
        clip_range = 0.075 #trial.suggest_float("clip_range", 0.05, 0.4, log=True)
        max_grad_norm = 0.5  #trial.suggest_float("max_grad_norm", 0.3, 3.0, log=True)
        ent_coef = 1e-6 #trial.suggest_float("ent_#coef", self.ent_coef_min, self.ent_coef_max, log=True)
        n_steps = 2 ** 14
        # trial.suggest_int("exponent_n_steps", self.exponent_n_steps_min, 
        # self.exponent_n_steps_max, log=True) # trying 10 and 14 instead of 10

        batch_size = 2**trial.suggest_int("batch_size", 9, 11, log=True)
        #batch_size = 2**trial.suggest_int("batch_size", 8, 10, log=True) #NOTE: 

        # batch_size = 2048 #NOTE update me
        #int(n_steps / 8) #int(n_steps / 8)
        # NOTE: I believe batch size of 8192 was causing the gpu to crash


        net_arch_width_int = trial.suggest_int("net_arch_width_int", 6, 8)
        # net_arch_width_int = trial.suggest_int("net_arch_width_int", 6, 8)
        net_arch_width = 2 ** net_arch_width_int
        net_arch_depth = trial.suggest_int("net_arch_depth", 3, 7)
        net_arch_array = np.ones(net_arch_depth,dtype=int)
        net_arch = [{"pi": (net_arch_array*net_arch_width).tolist(), "vf": (net_arch_array*net_arch_width).tolist()}]

        # Display true values not shown otherwise
        trial.set_user_attr("vf_coef", vf_coef)
        trial.set_user_attr("clip_range", clip_range)
        trial.set_user_attr("max_grad_norm", max_grad_norm)
        trial.set_user_attr("batch_size", batch_size)
        trial.set_user_attr("n_steps", n_steps)
        trial.set_user_attr("ent_coef", ent_coef)
        trial.set_user_attr("net_arch_width", net_arch_width)

        """ 
        NOTE: Rollout buffer size is number of steps*envs, should be in range of 2048-409600
        in this case we are testing only 1 enviroment at a time

        NOTE: Batch_size corresponds to how many experiences are used for each gradient descent update.
        his should always be a fraction of the buffer_size. If you are using a continuous action space,
        this value should be large (in 1000s). 
        """

        return {
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "max_grad_norm": max_grad_norm,
            "vf_coef": vf_coef,
            "clip_range": clip_range,
            "batch_size": batch_size, 
            "n_epochs": n_epochs,
            "policy_kwargs": {
                "net_arch": net_arch,
                "features_extractor_class": CustomCNN,
                "features_extractor_kwargs":dict(features_dim=1000),
            },
        }

    def objective(self, trial: optuna.Trial) -> float:

        objective_trial = Timer()
        objective_trial.start()

        kwargs = self.default_hyperparams.copy()
        # Sample hyperparameters

        kwargs.update(self.sample_ppo_params(trial))
        # Create the RL model

        print("\n \n")
        print(kwargs)


        # NOTE: Multi Enviroment
        n_envs = 3
        # n_envs = trial.suggest_int("n_envs", 3, 6, log=True) 
        # # trial.set_user_attr("n_envs", n_envs)
        # print("n_envs: ", n_envs)
        # print("\n \n")

        vec_env = make_env(env_id=self.env_id, n_envs=n_envs, seed=0, monitor_dir=self.logDir)
        vec_env = VecFrameStack(vec_env, n_stack=n_envs)

        model = PPO(env=vec_env, **kwargs)

        eval_callback = TrialEvalCallback(
            vec_env, trial, n_eval_episodes=self.n_eval_episodes, eval_freq=self.eval_freq, deterministic=True
        )

        nan_encountered = False

        try:
            # model.learn(total_timesteps = total_timesteps, callback=eval_callback)
            model.learn(total_timesteps = self.n_timesteps, callback=eval_callback)
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN
            print(e)
            nan_encountered = True
        finally:
            # Free memory
            model.env.close()
            # eval_env.close()

        # Tell the optimizer that the trial failed
        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        print("TRIAL Finished: Objective Trial Time")
        objective_trial.stop()

        return eval_callback.last_mean_reward
    

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 500, # EVALUATE MORE FREQUENTLY
        deterministic: bool = True,
        verbose: int = 2, # TRACKING THE RESULTS
        logDir = logDir,
        imgDir = imgDir,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.t_trial = Timer()
        self.t_trial.start()
        self.logDir = logDir
        self.imgDir = imgDir

        # New Entry
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1

            # Retrieve training reward
            x, y = results_plotter.ts2xy(results_plotter.load_results(self.logDir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-1000:])

            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            # self.trial.report(mean_reward, self.eval_idx)

            self.t_trial.stop() # end the timer for that round
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
            self.t_trial.start() #restart the timer
        return True


def Plotter(study):
    fig1 = plot_parallel_coordinate(study)
    fig2 = plot_optimization_history(study)
    fig3 = plot_param_importances(study)

    fig1.write_image(imgDir +"/plot_parallel_coordinate.jpeg")
    fig2.write_image(imgDir +"/plot_optimization_history.jpeg")
    fig3.write_image(imgDir +"/plot_param_importances.jpeg")


 
