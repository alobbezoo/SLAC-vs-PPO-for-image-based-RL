"""
Script for training SLAC with original hyperparameters

#import slac.registerRGB_PID_Curr
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
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        seed=args.seed,
        num_steps=args.num_steps, # task policy
        initial_learning_steps=args.initial_learning_steps, # latent representation
        initial_collection_steps= args.initial_collection_steps,
        eval_interval=args.eval_interval
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default= 6 * 10 ** 6) #default=2 * 10 ** 6) # for testing
    parser.add_argument("--initial_learning_steps", type=int, default= 2 * 10 ** 5) #default=2 * 10 ** 6) # for testing
    parser.add_argument("--eval_interval", type=int, default= 10 ** 4)
    parser.add_argument("--initial_collection_steps", type=int, default= 4*10 ** 4) # collection before training AC

    parser.add_argument("--domain_name", type=str, default="ArmGymJointControl")
    parser.add_argument("--task_name", type=str, default="reach")
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda", type=str, default="True") #cuda is true
    args = parser.parse_args()
    main(args)
