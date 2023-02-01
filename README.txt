This is a toy environment to rapidly test new machine learning approaches.

It serves two goals:
 1) Give you a glimpse of the types of problems we are solving (admittedly quite simplified in this toy interview setup to focus on core concepts).
 2) Give us a chance to see how you approach and solve problems in this area.

The base reinforcement learning algorithm implemented here is a bare-bones Deep Deterministic Policy Gradient (DDPG) \ref{https://arxiv.org/pdf/1509.02971.pdf}.
DDPG is an actor-critic architecture that allows model-free off-policy learning. It can efficiently handle continuous action spaces by using parameterized function approximators (via neural net models) to represent both the Q-function (critic) and the policy (actor). Iteratively, the actor acts like a policy network: given a state, it picks the action with (what it currently thinks yields) the highest expected reward. The critic model -- which acts like the Q-value network -- predicts the value of that action.

While this approach is pretty good and learns this task well in a few minutes, there are several aspects we can improve:

1. Besides DDPG, a Stochastic Latent Actor-Critic (SLAC) approach can be added, which offers a number of benefits over vanilla DDPG (\ref{https://arxiv.org/pdf/1907.00953.pdf} and possibly useful https://github.com/alexlee-gk/slac).
2. With SLAC, we can leverage machine vision to measure joints/hand/goal positions instead of simply reading them from the 2D environment. SLAC is well poised for this by learning a latent representation of complex signals (screen pixel values in this case). Right now the code "cheats" by always knowing where things are directly from code, but in reality such measures are hard to get and need to be estimated from noisy signals, such as video. Pixel frames can be easily saved/accessed with pyglet in the current setup.
4. Optimize architecture and hyperparameters to achieve more efficient learning, inference, faster convergence, model quality, and stability
5. Embed a PID-type control smoothing in the reward function to encourage smooth behavior and improve resistance to noise
6. Curriculum learning?
7. Anything else you think has an impact here? We are curious what you think.


Using any available papers, code, references, existing packages/libraries or github projects is fair game here. The goal is to get things working.

To install Dependencies:
	conda env create -f environment.yml

Or install manually:
	pip install tensorflow==1.15.4
	pip install pyglet (with pillow dependencies)


To run in inference mode: python main.py TEST
To run in learning mode: python main.py TRAIN

Note the API of the toy env.py is compatible with standard OpenAI gym environments. This makes plugging in existing ML libraries and visualization tools easy.


Feel free to reach out with questions at aim@machines.run

Happy coding!
