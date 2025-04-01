import json
import glob
import os
import pickle

from env.environment import Environment
from dqn.dqn_replay_buffer import ReplayBuffer, Transition
from dqn.dqn_networks import Linear

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np

from IPython import display
import scipy.ndimage as scind
from scipy.signal import savgol_filter

class DQN():
    def __init__(self, param_file='network_params', debug=False):
        if type(param_file) == str:
            with open(param_file, "r") as file:
                params = json.load(file)
                self.params = params
        else:
            self.params = param_file
            params = param_file

        if params['device'] == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.model = 'DQN'
        self.env = Environment(unit_number=params['unit_number'], incident_number=params['incident_number'],
                               obs_incident=params['obs_incident'],
                               event_per_day=params['event_per_day'], debug=debug, fast_T=params['fast_T'],
                               all_policies=params['all_policies'], penalty=params['invalid_penalty'],
                               time_format=params['time_format'],
                               fixed_units=params["fixed_units"], fixed_incidents=params["fixed_incidents"],
                               fixed_times=params["fixed_times"])
        self.n_actions = self.env.n_action
        self.n_observations = self.env.n_observation

        self.policy_net = Linear(self.n_observations, self.n_actions, hidden_layer=params['hidden_layer'],
                                 batch_norm=params['batch_norm']).to(self.device)
        self.target_net = Linear(self.n_observations, self.n_actions, hidden_layer=params['hidden_layer'],
                                 batch_norm=params['batch_norm']).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        if params['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=params['LR'], amsgrad=True,
                                         weight_decay=params['l2_lambda'] / (2 * params['LR']))
        elif params['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=params['LR'], amsgrad=True)
        elif params['optimizer'] == 'SGD' or True:
            self.optimizer = optim.SGD(self.policy_net.parameters(), lr=params['LR'])

        self.scheduler = StepLR(self.optimizer, step_size=params['LR_step_size'], gamma=params['LR_gamma'])
        self.memory = ReplayBuffer(self.params['MEMORY_SIZE'])

        self.steps_done = 0
        self.eps_threshold_pre = (self.params['EPS_START'] - self.params['EPS_END'])
        self.update_eps_threshold()
        self.invalid_last_action = False
        self.state = None

        self.plt_figsize = (10, 20)
        self.window_size = 100
        plt.ion()

        self.rewards_wp = []
        self.epsilons = []
        self.bad_actions = []
        self.buffer_capacity = []
        # self.illegal_buffer = []
        self.crit_losses = []
        self.losses = []
        self.lrs = []

        self.file_name = None

    def get_actions(self, x):
        self.policy_net.eval()
        with torch.no_grad():
            actions = self.policy_net(x)

        if self.params['soft_max']:
            soft_action = F.softmax(actions, dim=1)
            actions = torch.multinomial(soft_action, 1, replacement=True).squeeze(1)
        else:
            actions = actions.max(1).indices

        self.policy_net.train()
        return actions

    def reset(self):
        self.cum_reward = 0
        self.cum_reward_wp = 0
        self.bad_action_counter = 0
        self.state = self.env.reset()
        self.state = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def emulate(self):
        action = self.select_action()
        observation, reward, done = self.env.step(action.item())

        if reward == self.params['invalid_penalty']:
            self.bad_action_counter += 1
            self.invalid_last_action = True
        else:
            self.cum_reward_wp += reward

        self.cum_reward += reward

        reward = torch.tensor([reward], device=self.device)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Store the transition in memory
        self.memory.push(self.state, action, next_state, reward)

        # Move to the next state
        self.state = next_state
        return done

    def update_eps_threshold(self):
        self.eps_threshold = self.params['EPS_END'] + self.eps_threshold_pre * np.exp(
            -1 * self.steps_done / self.params['EPS_DECAY'])

        self.epsilons.append(self.eps_threshold)

    def select_action(self, evaluate=False):
        if evaluate or np.random.random() > self.eps_threshold and not self.invalid_last_action:
            with torch.no_grad():
                return self.policy_net(self.state).max(1).indices.view(1, 1)
        else:
            self.invalid_last_action = False
            return torch.tensor([[self.env.sample_action()]], device=self.device, dtype=torch.long)

    def update_target(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.params['TAU'] + target_net_state_dict[
                key] * (1 - self.params['TAU'])
        self.target_net.load_state_dict(target_net_state_dict)

    def optimize(self):
        if len(self.memory) < self.params['START_SIZE']:
            return

        self.steps_done += 1
        self.update_eps_threshold()
        transitions = self.memory.sample(self.params['BATCH_SIZE'])

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
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

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # state_batch.requires_grad_(True)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.params['BATCH_SIZE'], device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values

        expected_state_action_values = (next_state_values * self.params['GAMMA']) + reward_batch.squeeze(1)

        # Compute Huber loss
        if self.params['loss'] == 'mse':
            criterion = nn.MSELoss()
        elif self.params['loss'] == 'l1smooth':
            criterion = nn.SmoothL1Loss(beta=1000)
        else:
            criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        crit_loss = loss.item()

        if self.params['optimizer'] != 'AdamW':
            weight_decay_loss = 0
            for param in self.policy_net.parameters():
                weight_decay_loss += torch.norm(param, p=2) ** 2  # L2 norm
            weight_decay_loss /= len(list(self.policy_net.parameters()))
            weight_decay_loss *= self.params['l2_lambda']

            # print(loss)
            # print(weight_decay_loss)
            loss += weight_decay_loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        if self.scheduler.get_last_lr()[0] > self.params['min_LR']:
            self.scheduler.step()

        self.update_target()

        self.crit_losses.append(crit_loss)
        self.losses.append(loss.item())


    def transition_to_cpu(self, t):
        new_t = []
        for k in t:
            try:
                new_t.append(k.cpu())
            except:
                new_t.append(k)
        return Transition(*new_t)

    def transition_to_gpu(self, t):
        new_t = []
        for k in t:
            try:
                new_t.append(k.to(self.device))
            except:
                new_t.append(k)
        return Transition(*new_t)

    def save_model(self, episode=None):
        # Create a file name if it doesnt exists
        if self.file_name is None:
            file_name = [int(k.split('/')[1]) for k in glob.glob('models/*')]
            try:
                file_name = "models/" + str(np.max(file_name) + 1)
            except:
                file_name = "models/1"

            self.file_name = file_name
        try:
            os.mkdir(self.file_name)
        except:
            pass

        file_name = self.file_name
        if not episode is None:
            file_name += f'/{episode}'

            try:
                os.mkdir(file_name)
            except:
                pass

        # Save NN
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }, file_name + '/checkpoint.pth')

        # Save params
        with open(file_name + "/parameters", 'w') as f:
            json.dump(self.params, f)

        # Save statistics

        data = {
            "rewards_wp": self.rewards_wp,
            "epsilons": self.epsilons,
            "bad_actions": self.bad_actions,
            "buffer_capacity": self.buffer_capacity,
            "crit_losses": self.crit_losses,
            "losses": self.losses,
            "steps_done": self.steps_done,
            "file_name": self.file_name,
            "lrs": self.lrs
        }
        with open(file_name + "/data.pkl", 'wb') as f:
            pickle.dump(data, f)

        # Save memory
        memory_cpu = [self.transition_to_cpu(t) for t in dqn.memory.buffer]

        with open(file_name + '/memory.pkl', 'wb') as f:
            pickle.dump(memory_cpu, f)

    def load_model(self, file_name=None):
        if file_name is None:
            file_name = self.file_name
        if type(file_name) == int:
            file_name = self.file_name + '/' + str(file_name)
        # Load NN
        checkpoint = torch.load(file_name + '/checkpoint.pth')
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load params
        with open(file_name + "/parameters", 'r') as f:
            self.params = json.load(f)

        # Load statistics
        with open(file_name + "/data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.rewards_wp = data['rewards_wp']
            self.epsilons = data["epsilons"]
            self.bad_actions = data["bad_actions"]
            self.buffer_capacity = data["buffer_capacity"]
            self.steps_done = data['steps_done']
            self.file_name = data['file_name']
            self.crit_losses = data['crit_losses']
            self.losses = data['losses']
            self.lrs = data['lrs']

        self.update_eps_threshold()
        # Load memory
        with open(file_name + '/memory.pkl', 'rb') as f:
            memory_cpu = pickle.load(f)
        memory_gpu = [self.transition_to_gpu(t) for t in memory_cpu]
        self.memory.add(memory_gpu)

    def moving_average(data, window_size=25, mode='same', polyorder=3):
        # Define a window of ones with size equal to the window_size
        if mode == 'gauss':
            moving_avg = scind.gaussian_filter1d(data, sigma=(window_size - 1) / 6)
        elif mode == 'sg':
            moving_avg = savgol_filter(data, window_length=window_size, polyorder=polyorder)
        else:
            window = np.ones(window_size) / window_size
            # Compute the moving average using numpy.convolve
            moving_avg = np.convolve(data, window, mode=mode)
            # moving_avg = np.concatenate((np.zeros(window_size//2 ), moving_avg, np.zeros(window_size//2)))
        return moving_avg

    def plot_training(self, save=False, show=False):
        fig, axes = plt.subplots(5, figsize=self.plt_figsize)

        graphs = [self.bad_actions, self.rewards_wp, self.losses]
        names = ['Bad actions', 'Rewards', 'Loss']

        for graph, ax, name in zip(graphs, axes, names):
            ax.clear()

            ax.plot(graph)
            ax.set_ylabel(name)
            ax.grid(True)

            ax.plot(self.moving_average(graph, window_size=self.window_size, mode='gauss'))
            if "Loss" in name:
                ax.set_yscale('log')

        ax.set_xlabel('Episode')

        if save:
            plt.savefig(self.file_name + '/stats.png')
        plt.pause(0.001)  # pause a bit so that plots are updated

        if not show:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
