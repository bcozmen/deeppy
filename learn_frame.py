import random
from collections import deque, namedtuple
from data.envdata import EnvData
import matplotlib.pyplot as plt
import matplotlib
import torch
from IPython import display

class LearningFrame():
    def __init__(self, model, data, batch_size, start_size =None):



        self.is_ipython = 'inline' in matplotlib.get_backend()
            
        plt.ion()
        self.model=model
        self.data = data
        self.batch_size = batch_size
        self.start_size = start_size

        self.durations = []
        self.duration = 0
    def train(self, duration = True):
        self.model.train()
        if isinstance(self.data, EnvData):
            done = self.data.collect(self.model)
            self.duration +=1
            if done and duration:
                self.durations.append(self.duration)
                self.duration = 0
            if len(self.data) > self.start_size:
                X = self.data.train_data(self.batch_size)
                self.model.optimize(*X)
            return done

    def test(self):
        self.model.eval()
        if isinstance(self.data, EnvData):
            counter = 0
            self.data.reset()
            while(not self.data.collect(self.model)):
                counter += 1
            return counter

    def plot_durations(self,show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
    
        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())




