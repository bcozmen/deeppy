from deeppy.utils import print_args
from deeppy.data.envdata import EnvData
from deeppy.data.base import DatasetLoader


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

        self.colors = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
            'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
            'tab:olive', 'tab:cyan', 'tab:blueviolet', 'tab:gold',
            'tab:turquoise', 'tab:coral', 'tab:limegreen', 'tab:indigo'
        ]

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



    def plot_old(self, log= False, show_result = False, save = None):
        if self.env_data:
            data = [[self.reward_data], [self.duration_data]]
            labels = [["Reward"], ["Duration"]]
        else:
            data = [ [self.train_data, []]]
            labels = [[["Train " +lbl for lbl in self.train_labels], ["Test " +lbl for lbl in self.test_labels]]]
        if self.plot_lr:
            data += [self.lrs]
            labels += ["Learning Rate"]
        self.plot_data(data,labels,log=log,show_result=show_result,save=save)

    def plot(self, log = True, show_result = False, save = None, window_size = 12, show_lrs = False, text = ""):
        """
        Plot the training and test data.
        Parameters
        ----------
        log : bool, optional
            If True, plot the data on a logarithmic scale. The default is True.
        show_result : bool, optional
            If True, display the plot in the notebook. The default is False.
        save : str, optional
            If provided, save the plot to the specified file path. The default is None.
        window_size : int, optional
            The size of the window for smoothing the data. If None, no smoothing is applied.
        """
        num_rows = 1 + int(show_lrs)
        fig,axes = plt.subplots(num_rows, figsize = (12,10*num_rows))
        
        if num_rows == 1:
            axes = [axes]
        
        train_data = list(zip(*self.train_data))
        test_data =  list(zip(*self.test_data))

        ax = axes[0]
        print(ax)
        ax.set_xlabel('Steps')
        ax.clear()

        y_label = "Loss"
        if log:
            y_label = "Log10 Loss"
        ax.set_ylabel(y_label)
        ax.grid(True)
        
        for data,label,color in zip(train_data, self.train_labels,self.colors):
            ixes = self.train_data_ix
            if window_size is not None:
                data = self.uniform_smooth(data, window_size=window_size)
                ixes = ixes[window_size//2:-(window_size//2-1)]
            if log:
                data = np.log10(data)
    
            ax.plot(ixes,data, '-', c=color, label = "Train " +label , alpha=0.9, linewidth = 1.5)
                
        
        for data,label,color in zip(test_data, self.train_labels,self.colors):
            #if window_size is not None:
            #    data = self.gaussian_smooth(data, window_size=window_size)
            if log:
                data = np.log10(data)

            ax.plot(self.test_data_ix,data,'--', c = color, alpha=0.9)


        ax.legend()

        if show_lrs:
            ax = axes[1]
            ax.plot(self.train_data_ix,np.log10(self.lrs), label = "Learning Rate")
            ax.set_ylabel("Log10 Learning Rate")
            ax.set_xlabel("Epoch")
            ax.grid(True) 
        
        plt.xlabel('Steps')
        plt.title(text)
        if save is not None:
            plt.savefig(save)
        plt.pause(0.001)  # pause a bit so that plots are updated

        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
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

    def gaussian_smooth(self, data, window_size=5):
        """
        Apply Gaussian smoothing to the data.
        """
        if len(data) < window_size:
            return data
        kernel = np.exp(-np.linspace(-1, 1, window_size)**2 / 0.5**2)
        kernel /= kernel.sum()
        smoothed_data = np.convolve(data, kernel, mode='valid')
        return smoothed_data
    
    #implement uniform smoothing in a window
    def uniform_smooth(self, data, window_size=5):
        """
        Apply uniform smoothing to the data.
        """
        if len(data) < window_size:
            return data
        kernel = np.ones(window_size) / window_size
        smoothed_data = np.convolve(data, kernel, mode='valid')
        return smoothed_data





        
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

        if data == DatasetLoader:
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
        

    def test(self, steps = 1):
        losses = []
        for _ in range(steps):
            if isinstance(self.data, EnvData):
                loss = self.data.emulate(self.model)
                losses.append(loss)
                
            else:
                
                X = self.data.test_data()
                loss = self.model.test(X)
                losses.append(loss)
        
        loss = tuple(np.mean(list(zip(*losses)),axis=1))
                
        if isinstance(self.data, EnvData):
            self.metric.duration_data.append(loss[0])
            self.metric.reward_data.append(loss[1])
        else:
            self.metric.test_data.append(loss)
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
        self.optim_epoch = max(self.metric.train_data_ix) + 1


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

    def plot(self, show_result = True, log = False,save = None, show_lrs = False, window_size = 12, text=""):
        self.metric.plot(show_result=show_result, log=log,save=save,show_lrs=show_lrs, window_size=window_size, text = text)


