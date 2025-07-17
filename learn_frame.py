from deeppy.utils import print_args
from deeppy.data.envdata import EnvData
from deeppy.data.base import DatasetLoader


import torch
import random
import pickle


from collections import deque, namedtuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import trange
import os

        
class LearnFrame():
    print_args = classmethod(print_args)
    def __init__(self, model, data):
        self.model=model
        self.data = data


    def scheduler_step(self):
        self.model.scheduler_step()

    def reset(self):
        self.data.reset()

    def collect(self):
        self.model.train()
        done, reward = self.data.collect(self.model)
        return done

    def train(self,test_freq, epochs,gradient_accumulation_steps, path):
        for i in trange(self.model.optimizer._optimizer_steps_counter, epochs):

            self.optimize()

            if (i+1)%test_freq == 0:
                self.test(steps=gradient_accumulation_steps)
            
            if (i+1) % 10000 == 0:
                dire = path + f"{(i+1)}"
                
                try:
                    os.mkdir(dire)
                except:
                    pass
                self.save(dire)
        dire = path + "final"
        try:
            os.mkdir(dire)
        except:
            pass
        self.save(dire)

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
        optimizer_return = False

        while optimizer_return == False:
            #Get the next batch
            X = self.data.train_data()
            if X is None:
                return 
            
            optimizer_return = self.model.optimize(X)
        
    def test(self, steps = 1):
        self.model.eval()
        test_return = False
        
        for _ in range(steps):
            while test_return == False:
                X = self.data.test_data()
                test_return = self.model.test(X)
                
    def save(self, file_name, save_data = True):
        self.data.save(file_name)
        if save_data:
            self.model.save(file_name)

    def load(self, file_name, load_data = True):
        self.model = self.model.load(file_name)
        if load_data:
            self.data.load(file_name = file_name)


    def get_anim(self, name = None, interval = 100):
        frames = []
        self.data.reset()
        self.model.eval()
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


