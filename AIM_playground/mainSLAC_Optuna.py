"""
Script for tuning SLAC hyperparameters and training the best combination

Hyperparameters which were explored included: gamma, tau, lr_sac, lr_latent, batch batch_size_latent, batch_size_sac
Additionally feed forward NN width and depth were treated as hyperparameters
All other hyperparameters were fixed

"""

# Custom Imports
from slac.optunaHyp import OptunaFunc
from slac.optunaHyp import Plotter

from slac.optimize import TrainOpt


from tabnanny import verbose
import torch

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import registerRGB


#Training Constants
ENV_ID = "ArmGymJointControl-v1"

# Optuna
N_TRIALS = 10 #NOTE: Check and Update Me
N_EVAL_EPISODES = 10

# SLAC
N_STARTUP_TRIALS = 8
N_LATENT_STEPS = 10 ** 4 
N_LEARNING_STEPS = 10 **  5
N_INITIAL_COLLECTION_STEPS = 10 ** 4
EVAL_INTERVAL = 5 * 10 ** 3

OPTIMIZED_N_LATENT_STEPS = 4 * 10 ** 4 #NOTE: Check and Update Me
OPTIMIZED_N_LEARNING_STEPS = 8 * 10 ** 5
OPTIMIZED_N_INITIAL_COLLECTION_STEPS = 5 * 10 ** 3
OPTIMIZED_EVAL_INTERVAL = 10 ** 4


if __name__ == "__main__":
    
    optunaClass = OptunaFunc(
        n_latent_timesteps = N_LATENT_STEPS,        
        n_learning_timesteps = N_LEARNING_STEPS,
        n_init_collection_steps = N_INITIAL_COLLECTION_STEPS,
        eval_interval =EVAL_INTERVAL, 
        env_id = ENV_ID, 
    )

    objective = optunaClass.objective

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    except KeyboardInterrupt:
        pass

    #Plotting out the outputs
    Plotter(study)

    print("\n \n Number of finished trials: ", len(study.trials))

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

    #Training with the optimized hyperparameters 
    OptimizedStudy = TrainOpt(trial=trial, env_id=ENV_ID, n_latent_timesteps = OPTIMIZED_N_LATENT_STEPS,        
        n_learning_timesteps = OPTIMIZED_N_LEARNING_STEPS, initial_collection_steps = OPTIMIZED_N_INITIAL_COLLECTION_STEPS,
        eval_interval=OPTIMIZED_EVAL_INTERVAL)
    OptimizedStudy.train()


