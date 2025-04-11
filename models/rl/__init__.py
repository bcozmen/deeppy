from .dqn import DQN, DDQN
from .sac import SAC

algorithms =[a.__name__ for a in [DQN, DDQN,SAC]]
envs = ["CartPole-v1", "CartPole-v1", "LunarLander-v3"]
buffers = [10000,10000,100000]
classes = [DQN, DDQN,SAC]