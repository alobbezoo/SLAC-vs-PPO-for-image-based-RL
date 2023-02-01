"""
Script for tuning PPO hyperparameters and training the best combination

Hyperparameters which were explored included: gamma, gae_lambda, learning_rate, n_epochs, batch size
Additionally feed forward NN width and depth were treated as hyperparameters
vf_coef, clip_range, max_grad_norm, ent_coef, n_steps, were all fixed

"""

# Custom Imports
from ppo.optunaHyp import OptunaFunc, TrialEvalCallback
from ppo.optunaHyp import Plotter
from ppo.customCnnShallow import CustomCNN
from ppo.optimize import TrainOpt
import registerRGB

from tabnanny import verbose

import torch

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

#Training Constants
N_TRIALS = 30
N_STARTUP_TRIALS = 8
N_EVALUATIONS = 10
N_TIMESTEPS = 1e5
EVAL_FREQ = 1e3
N_EVAL_EPISODES = 10
RENDER = False
VERBOSE = 2

# Hyperparameters of high importance to Tune

LEARNING_RATE_MIN = 5e-4
LEARNING_RATE_MAX = 5e-3
ENT_COEF_MIN = 1e-8 
ENT_COEF_MAX = 1e-1

ENV_ID = "ArmGymJointControl-v1"
DEVICE = "cpu" #"cuda"


OPTIMIZED_N_TIMESTEPS = 10e6

optunaClass = OptunaFunc(n_timesteps = N_TIMESTEPS, eval_freq =EVAL_FREQ, 
        n_eval_episodes = N_EVAL_EPISODES, env_id = ENV_ID, 
        learning_rate_min = LEARNING_RATE_MIN, learning_rate_max = LEARNING_RATE_MAX, 
        ent_coef_min = ENT_COEF_MIN, ent_coef_max = ENT_COEF_MAX,
        policy = "CnnPolicy", device=DEVICE, render = RENDER, verbose = VERBOSE,
    )

objective = optunaClass.objective


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training
    # torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_TIMESTEPS // 3)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    except KeyboardInterrupt:
        pass

    #Plotting out the outputs
    Plotter(study)

    print("\n \nNumber of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
    
    print("\n \n \n \n")
    print("-----------------------------------------------------------------------------------")
    print("TRAINING OPTIMIZED STUDY")
    print("\n \n \n \n")
