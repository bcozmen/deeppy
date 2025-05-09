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




    
class Metric():
    def __init__(self,train_labels,test_labels, env_data=False):
        #Labels list of strings for plot labels
        self.env_data = env_data
        self.plot_lr = False

        self.is_ipython = 'inline' in matplotlib.get_backend()
        plt.ion()

        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_data,self.test_data = [], []
        self.train_data_ix,self.test_data_ix = [], []

        self.reward, self.duration = 0,0
        self.duration_data, self.reward_data = [],[]
        
        self.lrs = []

    def env_reset(self):
        self.reward, self.duration = 0,0

    def env_done(self):
        self.duration_data.append(self.duration)
        self.reward_data.append(self.reward)
        self.env_reset()
    def env_step(self,reward, done):
        self.duration +=1
        self.reward += reward
        if done:
            self.env_done()



    def plot(self, log= False, show_result = False, save = None):
        if self.env_data:
            data = [[self.reward_data], [self.duration_data]]
            labels = [["Reward"], ["Duration"]]
        else:
            data = [ [self.train_data, self.test_data]]
            labels = [ [["Train " +lbl for lbl in self.train_labels], ["Test " +lbl for lbl in self.test_labels]]]
        if self.plot_lr:
            data += [self.lrs]
            labels += ["Learning Rate"]
        self.plot_data(data,labels,log=log,show_result=show_result,save=save)


    def plot_ax(self,ax,data, label, log = False):
        ax.set_xlabel('Episode')
        ax.clear()
        #ax.set_ylabel(name)
        ax.grid(True)

        
        is_lr = (label == "Learning Rate")

        if is_lr:
            data = np.log10([data])
            label = np.asarray([label])

        for d,l in zip(data,label):
            d = np.asarray(d)
            if log and not is_lr:
                d = np.log10(d)
            ax.plot(d,label=l)
        ax.legend()

    def plot_data(self,datas,labels, log=False,show_result=False, save = None):
        num_rows = len(labels)
        fig,axes = plt.subplots(num_rows, figsize = (10,6*num_rows))
        if num_rows == 1:
            axes = [axes]


        plt.xlabel('Episode')


        for data,label,ax in zip(datas,labels,axes):
            self.plot_ax(ax,data,label, log = log)
        
        if save is not None:
            plt.savefig(save)
        plt.pause(0.001)  # pause a bit so that plots are updated


        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())





        
class LearnFrame():
    print_args = classmethod(print_args)
    def __init__(self, model, data):
        self.optim_epoch = 0
        self.model=model
        self.return_labels = self.model.optimize_return_labels

        try:
            self.test_return_labels = self.model.test_return_labels
        except:
            self.test_return_labels = self.return_labels
        env_data = False
        if isinstance(data, EnvData):
            env_data = True
        self.metric = Metric(self.return_labels, self.test_return_labels, env_data = env_data)

        if data == DataGetter:
            data = data()
        self.data = data


    def scheduler_step(self):
        self.model.scheduler_step()

    def reset(self):
        self.data.reset()
        self.metric.env_reset()

    def collect(self):
        self.model.train()
        done, reward = self.data.collect(self.model)
        self.metric.env_step(reward,done)
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
        
        train = self.model.optimize(X)
        self.metric.train_data.append(train)
        self.metric.train_data_ix.append(self.optim_epoch)
        self.optim_epoch += 1
        try:
            self.metric.lrs.append(self.model.last_lr()[0])
        except:
            pass
        return train
        

    def test(self):
        self.model.eval()
        if isinstance(self.data, EnvData):
            dur, rew = self.data.emulate(self.model)
            self.metric.duration_data.append(dur)
            self.metric.reward_data.append(rew)
        else:
            X = self.data.test_data()
            test = self.model.test(X)
            self.metric.test_data.append(test)
            self.metric.test_data_ix.append(self.optim_epoch)

    def save(self, file_name, save_data = True):
        self.data.save(file_name)
        if save_data:
            self.model.save(file_name)

        with open(file_name + "/data.pkl", 'wb') as f:
            pickle.dump({
                'metric': self.metric,
            }, f)

    def load(self, file_name, load_data = True):
        self.model = self.model.load(file_name)
        if load_data:
            self.data.load(file_name = file_name)
        with open(file_name + "/data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.metric = data['metric']


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

    def plot(self, show_result = False, log = False,save = None):
        self.metric.plot(show_result=show_result, log=log,save=save)







