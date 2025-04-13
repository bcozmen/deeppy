from .dqn import DQN, DDQN
from .sac import SAC

algorithms =[a.__name__ for a in [DQN, DDQN]] + ["SAC_Discrete" , "SAC_Continuous"]
envs = ["CartPole-v1", "CartPole-v1", "LunarLander-v3", "HalfCheetah-v5"]
buffers = [10000,10000,100000, 1e6]
classes = [DQN, DDQN,SAC, SAC]