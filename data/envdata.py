import random
from collections import deque, namedtuple

from deeppy.data.base import Base

import torch
import pickle

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','done'))

class ReplayBuffer():
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)
    def __len__(self):
        return len(self.buffer)
    def __getitem__(self, index):   
        return self.buffer[index]
    def push(self, args):
        self.buffer.extend([Transition(*k) for k in zip(*args)])

    def sample(self, batch_size):
        return [torch.stack(k) for k in (zip(*random.sample(self.buffer,batch_size)))]

class EnvData(Base):
    def __init__(self, env, buffer_size = 20000, batch_size = 128, start_size = 128):
        super().__init__( batch_size = batch_size)
        self.memory = ReplayBuffer(buffer_size)
        self.env = env

        try:
            self.env.action_space.n
            self.action_item = True
        except:
            self.action_item = False

        self.start_size = start_size
        self.reset()

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.state = torch.tensor(self.env.reset()[0], dtype=torch.float32).unsqueeze(0)
    
    def train_data(self):
        if len(self.memory) < self.start_size:
            return None
        return  self.memory.sample(self.batch_size)

    def collect(self, model):
        action = model.predict(self.state).to(self.device)

        env_action = action.squeeze(0).numpy()
        if self.action_item:
            env_action = env_action.item()
        observation, reward, termination, truncation, data = self.env.step(env_action)

        done = torch.tensor([(termination or truncation)], dtype=torch.bool).unsqueeze(0)
        reward = torch.tensor([reward],dtype = torch.float32).unsqueeze(0)
        next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        # Store the transition in memory
        if model.training:    
            self.memory.push((self.state, action, next_state, reward,done))

        # Move to the next state
        self.state = next_state
        if done.item():
            self.reset()
        return done.item(), reward.item()

    def save(self,file_name):
        with open(file_name + '/memory.pkl', 'wb') as f:
            pickle.dump(self.memory.buffer, f)

    def load(self, file_name):
        with open(file_name + '/memory.pkl', 'rb') as f:
            self.memory.buffer = pickle.load(f)

