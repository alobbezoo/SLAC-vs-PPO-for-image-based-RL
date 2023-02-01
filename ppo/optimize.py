#Grasping Task Optimization: 

#Custom Imports
from ppo.optunaHyp import make_env
from ppo.customCnnShallow import CustomCNN

#Standard Imports
import gym
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback

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

pathDir = homeDir + "ppo/" + str(date) + "/"
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
        self, trial, env_ID, timeSteps, policy = "CnnPolicy", device="cpu", verbose=2, render = False, 
        eval_freq=10000, pathDir = pathDir, imgDir=imgDir, logDir=logDir
    ):
        self.eval_freq = eval_freq
        self.trial = trial
        self.env_ID = env_ID
        self.timeSteps = timeSteps
        self.policy = policy
        self.device = device
        self.verbose = verbose
        # self.policy_kwargs = policy_kwargs
        self.render = render
        self.pathDir = pathDir
        self.logDir = logDir
        self.imgDir = imgDir


    def optimal(self):
        learning_rate = self.trial.params['lr']
        gae_lambda = self.trial.params['gae_lambda']
        n_epochs = self.trial.params['n_epochs']
        gamma = self.trial.params['gamma']
        net_arch_depth = self.trial.params['net_arch_depth']
        

        vf_coef = self.trial.user_attrs['vf_coef']
        clip_range = self.trial.user_attrs['clip_range']
        max_grad_norm = self.trial.user_attrs['max_grad_norm']
        batch_size = self.trial.user_attrs['batch_size']
        n_steps = self.trial.user_attrs['n_steps']
        ent_coef = self.trial.user_attrs['ent_coef']
        net_arch_width = self.trial.user_attrs['net_arch_width']

        net_arch_array = np.ones(net_arch_depth,dtype=int)
        net_arch = [{"pi": (net_arch_array*net_arch_width).tolist(), "vf": (net_arch_array*net_arch_width).tolist()}]

        self.policy_kwargs = {
            "net_arch": net_arch,
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs":dict(features_dim=1000),
            }

        self.n_envs = 3 #self.trial.params['n_envs']

        kwargs = {
            "policy": self.policy, 
            "device": self.device,
            "verbose": self.verbose, 
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "clip_range": clip_range,
            "max_grad_norm": max_grad_norm,
            "batch_size": batch_size, 
            "n_steps": n_steps, 
            "n_epochs": n_epochs, 
            "policy_kwargs": self.policy_kwargs,
        }

        return kwargs

    def train(self):

        kwargs = self.optimal()
        print("\n \nKwargs are: ")
        print(kwargs)

        
        self.vec_env = make_env(env_id=self.env_ID, n_envs=self.n_envs, seed=0, monitor_dir=self.logDir)
        vec_env = VecFrameStack(self.vec_env, n_stack=self.n_envs)
        

        callback = SaveOnBestTrainingRewardCallback(check_freq=self.eval_freq, log_dir=self.logDir, verbose=self.verbose)

        self.model= PPO(env=self.vec_env, **kwargs)

        self.model.learn(total_timesteps=int(self.timeSteps), callback=callback)

        results_plotter.plot_results([self.logDir], self.timeSteps, results_plotter.X_TIMESTEPS, "PPO PandaReach")
        plt.savefig(str(self.imgDir) +'/optimize_plot.png')
    

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1, modelDir = modelDir):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.modelDir = modelDir

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = results_plotter.ts2xy(results_plotter.load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward

                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    model_path = self.modelDir + "model_" + str(timestamp)
                    self.model.save(path=model_path)

                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True
