import registerRGB
import gym
import numpy as np

env = gym.make("ArmGymJointControl-v1")

observation = env.reset()

sumReward = []
steps = 3000
for _ in range(steps):
   observation, reward, done, info = env.step(env.action_space.sample())
   sumReward.append(reward)

   if done:
      observation = env.reset()

print("sum is: ", np.sum(sumReward)/(steps/300))

# Output usually ~-150
