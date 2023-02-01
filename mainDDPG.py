import os
import sys
import pyglet
from ddpg.env import ArmEnv
from ddpg.rl_v2 import DDPG

import numpy as np
import time


MAX_EPISODES = 2500
MAX_EP_STEPS = 300
ON_TRAIN = sys.argv[1] == 'TRAIN'
CONTINUE_TRAINING = False

# Disable GPU computation
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []


def on_key_press(symbol, _):
    if symbol == pyglet.window.key.ESCAPE:
        env.viewer.close()


def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            env.render()
            a = rl.choose_action(s)
            # print("action is: ", a)
            s_, r, done = env.step(a)
            # print("state is: ", s_)
            # time.sleep(1)
            rl.store_transition(s, a, r, s_)
            ep_r += r
            
            if rl.memory_full:
                # start to learn once memory is full
                rl.learn()
            s = s_
            if done or j == MAX_EP_STEPS - 1:
                print('Ep: %i | %s | ep_r: % 6.1f | step: %i' % (i, '----' if not done else 'done', ep_r, j))
                break

        # print("max value is: ", np.asarray(rl.action_array).max(axis=0))
        # print("min value is: ", np.asarray(rl.action_array).min(axis=0))

        if i % 50 == 0 and i > 0:
            rl.save()
    rl.save()


def eval():
    rl.restore()
    env.render()
    window = env.viewer
    window.set_vsync(True)
    window.push_handlers(on_key_press)
    cursor = window.get_system_mouse_cursor(pyglet.window.Window.CURSOR_HAND)
    window.set_mouse_cursor(cursor)
    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done = env.step(a)


if ON_TRAIN:
    if CONTINUE_TRAINING:
        rl.restore()
    train()
else:
    eval()
