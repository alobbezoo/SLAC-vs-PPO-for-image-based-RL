INSTALLATIONS:

In addition to the environment.yml file, the environment.txt file contains packages that will be required.
You can install these python libraries in your existing conda environment using `pip3 install -r requirements.txt`.

If you're using CUDA, please install PyTorch following [instructions](https://pytorch.org/get-started/locally/) here.



FILE MANAGEMENT: 

All "main" scripts inside the parent directory can be used to initiate various training methods. Each script contains a short discription of its purpose. 
The plot_results folder contains the results from hyperparameter tuning and the optimization histories for PPO and SLAC. The PPO example was run with tuned hyperparameters and the original enviroment (no PID reward shaping and curriculum training). The SLAC example was run with original hyperparameters, action repeat of 1, and the original enviroment (no PID reward shaping and curriculum training). 



1,2) SLAC

The SLAC agent implemented for this problem was based on the github release (https://github.com/ku2482/slac.pytorch) uploaded by Toshki Watanabe  and Jan Schneider. This codebase implements an SLAC Pytorch  model to train gym-Mujoco Walker and Cheetah environments.

To accommodate this codebase, I created the custom gym environment called "ArmGymJointControl" which can be found in the registerRGB.py file. The gym environment was built based on a modified version of the ArmEnv provided. The step and reset methods in the gym environment calls a custom render method I wrote inside the ArmEnv Viewer class.

The standard pyglet render function was not working initially due to incorrect Linux OS config settings in the ArmEnv constructor. I modified the constructor, then was able to save, open, and compress the images using PIL. The render method of the ArmEnv Viewer Class returns the image as a 3x64x64 state representation for SLAC (and PPO) training.

The SLAC pytorch model showed excellent results for training the Mujoco cheetah model. 100 thousand latent training samples and two-million exploratory samples were all that was required to complete training and receive accurate results. I believe these results were achieved in part due to some hyperparameter tuning which were completed by the authors of the SLAC paper.

Testing the algorithm directly on the ArmGymJointControl environment resulted in poor (seemingly unconverging) performance. To verify that the environment was correctly set up, I implemented the stable-baselines3 PPO codebase (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html). The PPO algorithm was able to achieve better convergence of rewards, averaging a return of -30 per episode with no hyperparameter tuning.



3) HYPERPARAMETER TUNING

After setting up the gym environment, and implementing the SLAC and PPO algorithms, I went about the process of hyperparameter tuning. To save time testing combinations of hyperparameters, I implemented Optuna (https://optuna.readthedocs.io/en/stable/reference/optuna.html) a hyperparameter tuning package. I chose the basic TPESampler (Guassian Mixture Model) for selecting hyperparameter combinations.

The vision based ArmGymJointControl environment was slow, due to the time required for image saving, loading and compression. Testing hyperparameter combinations required significant computing power and clock-time. 

For SLAC, I was only able to test 20 combinations of hyperparameters. From the optimal hyperparameter combination from the 20 samples, I did not see significant improvement relative to the original. This could indicate that the initial hyperparameters were well selected, or the hyperparameter ranges which I suggested for exploration were poor. The one change I tested which did improve the learning for SLAC was to action repeat. Action repeat for the cheetah model was set to 4 but I found with an action repeat of 1, the model converged more quickly.

I set up the PPO model to allow for parallelization. The parallelization sped up the training time, which meant I was able to test 30 combinations of hyperparameters, and to run each hyperparameter combination for more timesteps. 30 combinations is not enough to find the ideal hyperparameter combination, but I did find that hyperparameter exploration significantly improved the training. The ideal hyperparameter set achieved an average return of +58 after ~5.4 million exploratory steps, which was significantly superior to the untuned model.



4) PID REWARD SHAPING

After completing hyperparameter tuning, I tested PID reward shaping for the environment. For this problem, the integral (I) term can be seen as the steady state distance between the target position and the end effector. The reward from the proportional (P) term can be viewed as the reward for the velocity of the end effector. The final derivative (D) term can be seen as the reward related to the acceleration of the end effector.

The integral term is incorporated into the original reward function. The agent receives a negative reward proportional to the distance between the end effector and the target. The agent receives a positive reward if that distance is zero. To improve behavior for stability and smoothness, P and D were added. 

For a real system, the end effector would have a maximum velocity constraint. I imposed this constraint by adding a negative reward which is fed to the agent if the end effector velocity exceeds a value of 80 pixels/second. The value of the negative reward is proportional to the magnitude by which the velocity exceeds 80 pixel/second.

For a real system, each joint would also have an acceleration limit. To meet this limit, and to prevent “jerking” as seen in the agent with the original reward function, I added a negative reward which is applied if the end effector acceleration is greater then 300 pixels/second^2. The value of the negative reward for acceleration is proportional to the magnitude by which the acceleration exceeds 300 pixels/second^2.

In the real world, the acceleration and velocity constraints should be added for the joints, and not for the end effector. For simplicity with this simulated reaching task, I prioritised the smoothness with which the end effector would move.

The results of this reward shaping, can be noticed in the  smoother motion seen during training for both PPO and DDPG. The overall rewards received by the agent per epoch are decreased with this type of reward function,  since negative rewards are being added for high acceleration, and no positive rewards are added for low acceleration.

This form of reward shaping limits the agent during exploration. The  reward signal is more information dense, which complicates the learning process. Additional training time is required for convergence with this reward function. 

 

5) CURRICULUM LEARNING

The final modification to the environment was curriculum learning. The goal of curriculum learning is to teach the agent to solve simple-sub problems, before solving the problem with full complexity.

From the agents perspective, there are 2 difficulties with this task. The first difficulty is that the block is sometimes far away from the agent, so positive rewards can not be achieved. For episodes where this is the case, the agent will have a difficult time finding the optimal position, since no matter how much the agent explores it will aways have negative rewards. The second difficulty is the overlap which can occur with the block and the arm. If the block is underneath the arm (near the base of the arm), the arm may obscure parts of the target block.

With the curriculum learning environment, the task is split into 3 stages. During the first stage (the first 500k steps), the target block is instantiated in a random position with a distance of 150 pixels (radius) from the center joint of the agent. This position is reachable, and the block and arm geometries are clearly distinct. In the second stage (500K-1M steps) the block will be instantiated anywhere inside the 150-pixel (radius) circular area surrounding the gripper. For this case, the agent must learn how to deal with the overlap between the arm and the target block. After 1M steps the block can be instantiated anywhere in the 400x400 pixel frame.

Note, the section above describes the curriculum learning environment for PPO, the SLAC stages were slightly different.

The results from curriculum learning were mixed. The agent achieves high results quickly in the first 500k steps, however each time the task difficulty increases, the rewards return to random levels (~-150).  Adding further stages in the curriculum model may be required, to make the transfer of knowledge between the stages smoother.

 

DISCUSSION

By implementing image-based training with SLAC and PPO, I was able the agent to reach the target block consistently with high positive rewards. That said, the rewards achieved for both models were significantly lower then DDPG with vector-based position represent. I believe with further hyperparameter tuning and training time I would be able to implement these two models and achieve a significant improvement.

Future steps which I would like to test, would be to:
1)     Modify the SLAC agent for parallelization (the agent gathers samples very slowly)
2)     Hyperparameter tuning (at least 100 hyperparameter combinations for both PPO and SLAC)



SLAC Vs PPO:

SLAC seems to have a much higher sample efficiency, as it took ~1 million steps to achieve positive rewards, compared to PPO which required ~3 million steps. In this case, I was able to use PPO parallelization to speed up the clocktime for training, however once I incorporate parallelization for SLAC I expect PPO to be significantly outperformed.

 
