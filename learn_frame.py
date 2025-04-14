from deeppy.utils import print_args
from deeppy.data.envdata import EnvData
from deeppy.data.dataset import DataGetter


import torch
import random
import pickle


from collections import deque, namedtuple

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from IPython import display
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np

class LearnFrame():
    print_args = classmethod(print_args)
    def __init__(self, model, data):
        self.is_ipython = 'inline' in matplotlib.get_backend()
            
        plt.ion()
        self.model=model
        if data == DataGetter:
            data = data()
        self.data = data


        self.duration = 0
        self.reward=0

        self.train_data = []
        self.test_data = []

        self.duration_data = []
        self.reward_data = []
        self.lrs = []

    def scheduler_step(self):
        self.model.scheduler_step()

    def reset(self):
        self.duration = 0
        self.reward = 0
        self.data.reset()
    def collect(self):
        self.model.train()

        try:
            self.data.collect
        except:
            raise TypeError("Data has no collect")

        done, reward = self.data.collect(self.model)
        self.duration +=1
        self.reward += reward



        if done:
            self.duration_data.append(self.duration)
            self.reward_data.append(self.reward)

            self.duration = 0
            self.reward = 0

        return done

    def optimize(self):
        """
        Gets training data from data, and trains the algorithms one step. 
        Parameters
        ----------

        Returns
        -------
        loss
            Loss objects (shape depends on the algorithm)
        """
        self.model.train()
        X = self.data.train_data()

        #For RL models, if start_size is not reached
        if X is None:
            return None

        r = self.model.optimize(*X)
        self.train_data.append(r)
        self.lrs.append(self.model.last_lr())

        return r
        

    def test(self):
        self.model.eval()
        if isinstance(self.data, EnvData):
            counter = 0
            cum_reward = 0

            self.data.reset()
            done = False
            while(not done):
                done = self.data.collect(self.model)
                counter += 1
                cum_reward += 1
        else:
            X = self.data.test_data()
            r = self.model.test(*X)
            self.test_data.append(r)

    def save(self, file_name):
        self.data.save(file_name)
        self.model.save(file_name)

        with open(file_name + "/data.pkl", 'wb') as f:
            pickle.dump({
                'train_data': self.train_data,
                'test_data': self.test_data,
                'duration_data': self.duration_data,
                'reward_data': self.reward_data,
            }, f)

    def load(self, file_name, load_data = True):
        self.model = self.model.load(file_name)
        if load_data:
            self.data.load(file_name = file_name)
        with open(file_name + "/data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.train_data = data['train_data']
            self.test_data = data['test_data']
            self.duration_data = data['duration_data']
            self.reward_data = data['reward_data']


    def get_anim(self, name = None, interval = 100):
        frames = []
        self.data.reset()
        
        frames.append(self.data.env.render())
        done = False
        while(not done):
            done, reward = self.data.collect(self.model)
            frames.append(self.data.env.render())
 

        fig, ax = plt.subplots()
        def animate(t):
            ax.cla()
            ax.imshow(frames[t])

        anim = FuncAnimation(fig, animate, frames=len(frames), interval = interval)

        if name is None:
            return anim
        fig.suptitle(name, fontsize=14) 
          
        # saving to m4 using ffmpeg writer 
        writervideo = animation.FFMpegWriter(fps=60) 
        anim.save(name + ".mp4", writer=writervideo) 
        plt.close()

    def plot(self,datas, labels,show_result=False, save = None):
        #datas = [[1,2,3]]

        num_rows = len(datas)
        fig,axes = plt.subplots(num_rows, figsize = (10,6*num_rows))
        if num_rows == 1:
            axes = [axes]


        plt.xlabel('Episode')


        for data,label,ax in zip(datas,labels,axes):
            ax.clear()
            #ax.set_ylabel(name)
            ax.grid(True)
            data = np.asarray(data)

            if len(data.shape) == 1:
                ax.plot(data, label = label)
            else:
                ax.plot(data.T, label = label)
        
            ax.legend()
        if save is not None:
            plt.savefig(save)
        plt.pause(0.001)  # pause a bit so that plots are updated


        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())




