"""
Script for training SLAC after ideal hyperparameters were found with optuna

# import slac.registerRGB_PID_Curr
Import this enviroment (line 18) if testing with PID reward shaping and curriculum learning

"""

import argparse
import os
from datetime import datetime

import torch

from slac.algo import SlacAlgorithm
from slac.trainer import Trainer

import registerRGB
import gym

def main(args):
    env = gym.make("ArmGymJointControl-v1")
    setattr(env, 'action_repeat', args.action_repeat)

    env_test = gym.make("ArmGymJointControl-v1")
    setattr(env_test, 'action_repeat', args.action_repeat)

    log_dir = os.path.join(
        "logs",
        f"{args.domain_name}-{args.task_name}",
        f'slac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}',
    )

    algo = SlacAlgorithm(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_repeat=args.action_repeat,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        gamma=args.gamma,
        batch_size_sac=args.batch_size_sac,
        batch_size_latent=args.batch_size_latent,
        hidden_units=args.hidden_units,
        lr_sac=args.lr_sac,
        lr_latent=args.lr_latent,
        tau=args.tau,

    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        seed=args.seed,
        num_steps=args.num_steps, # task policy
        initial_learning_steps=args.initial_learning_steps, # latent representation
        eval_interval=args.eval_interval,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default= 4 * 10 ** 6) #default=2 * 10 ** 6) # for testing
    parser.add_argument("--initial_learning_steps", type=int, default= 5 * 10 ** 5) #default=2 * 10 ** 6) # for testing
    parser.add_argument("--domain_name", type=str, default="ArmGymJointControl")
    parser.add_argument("--task_name", type=str, default="reach")
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda", type=str, default="True") #cuda is true

    # New Arguments:
    parser.add_argument("--eval_interval", type=int, default= 5 * 10 ** 3)
    parser.add_argument("--batch_size_sac", type=int, default= 512) #default=2 * 10 ** 6) # for testing
    parser.add_argument("--batch_size_latent", type=int, default= 32) #default=2 * 10 ** 6) # for testing
    parser.add_argument("--hidden_units", type=tuple, default=(128, 128, 128))
    parser.add_argument("--lr_sac", type=float, default= 4.234955262626269e-06) #default=2 * 10 ** 6) # for testing
    parser.add_argument("--lr_latent", type=float, default= 0.0006748075658097397) #default=2 * 10 ** 6) # for testing
    parser.add_argument("--tau", type=float, default=0.006731177811510084)
    parser.add_argument("--gamma", type=float, default=0.9534396047040001)


    args = parser.parse_args()
    main(args)
