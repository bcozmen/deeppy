import random
from collections import deque, namedtuple

from networks.FFN import FFN
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer():
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer_object = Transition

        self.buffer = deque(maxlen=capacity)
        self.illegal_count = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(Transition(*args))
    def add(self, items):
        if not type(items) is list:
            items = [items]
        if not all(isinstance(element, self.buffer_object) for element in items):
            raise ValueError('You have tried to add an invalid objects to the buffer')
        self.buffer.extend(items)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class LearningFrame():
    def __init__(self, model, data, batch_size, start_size =None):
        self.model=model
        self.data = data
        self.batch_size = batch_size
        self.start_size = start_size
    def train(self):
        if isinstance(self.data, EnvData):
            done = self.data.collect(self.model)
            if len(self.data) > self.start_size:
                X = self.data.train(self.batch_size)
                self.model.train(*X)
                return done

    def test(self):
        if isinstance(self.data, EnvData):
            counter = 0
            self.data.reset()
            while(not self.data.collect(self.model)):
                counter += 1
            return counter

class EnvData():
    def __init__(self, env, buffer_size,  device = None):
        self.memory = ReplayBuffer(buffer_size)
        self.env = env
        self.device = device

        self.reset()
    def reset(self):
        self.state = self.env.reset()[0]
        self.state = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
    def __len__(self):
        return len(self.memory)
    def train(self, batch_size):
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        return state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states
    
    def collect(self, model):
        action = model.predict(self.state, self.env.action_space.sample)
        observation, reward, termination, truncation, data = self.env.step(action.item())

        done = (termination or truncation)

        reward = torch.tensor([reward], device=self.device).unsqueeze(0)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Store the transition in memory
        self.memory.push(self.state, action, next_state, reward)

        # Move to the next state
        self.state = next_state
        if done:
            self.reset()
        return done


class DQN():
    def __init__(self, network_params, device = None, criterion = nn.MSELoss, 
                 gamma = 0.99, tau = 0.005,
                eps_start = 0.9, eps_end = 0.05, eps_decay = 1000):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.steps_done = 0
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        network_params["device"] = device

        
        self.tau = tau
        self.gamma = gamma
        
        self.criterion = criterion()
        
        self.policy_net = FFN(**network_params)
        self.target_net = FFN(**network_params)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for param in self.target_net.parameters():
             param.requires_grad = False
        

    def update_eps(self):
        self.steps_done += 1
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.steps_done / self.eps_decay)
    def update_target(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[
                key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
    
    def predict(self, x, random, eval= False):
        self.update_eps()
        self.policy_net.eval()

        if np.random.rand() > self.eps_threshold:
            with torch.no_grad():
                actions = self.policy_net.model(x)
                actions = actions.max(1).indices.view(1,-1)
        else:
            actions = torch.tensor([[random()]], device=self.device, dtype=torch.long)
        return actions
    
    def train(self, state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states):        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(len(state_batch), device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.squeeze(1)

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        crit_loss = loss.item()

        # Optimize the model
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()
        
        self.update_target()

