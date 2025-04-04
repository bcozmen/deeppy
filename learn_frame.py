import random
from collections import deque, namedtuple
from data.envdata import EnvData
import matplotlib.pyplot as plt
import matplotlib
import torch
from IPython import display
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

class LearningFrame():
    def __init__(self, model, data):



        self.is_ipython = 'inline' in matplotlib.get_backend()
            
        plt.ion()
        self.model=model
        self.data = data

        metrics = []


        self.rewards = []
        self.reward = 0

        self.durations = []
        self.duration = 0

        self.train_loss = [[],[],[]]
        self.test_loss = [[],[],[]]
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
            self.durations.append(self.duration)
            self.duration = 0

            self.rewards.append(self.reward)
            self.reward = 0

    def train(self):
        self.model.train()
        X = self.data.train_data()

        #For RL models, if start_size is not reached
        if X is None:
            return 0
        loss, mse, kl = self.model.optimize(*X)
        self.train_loss[0].append(loss)
        self.train_loss[1].append(mse)
        self.train_loss[2].append(kl)
        return loss

    def test(self):
        self.model.eval()
        if isinstance(self.data, EnvData):
            counter = 0
            cum_reward = 0

            self.data.reset()
            done = False
            while(not done):
                done, reward = self.data.collect(self.model)
                counter += 1
                cum_reward += reward
            return counter, cum_reward
        else:
            X = self.data.test_data()
            loss, mse, kl = self.model.test(*X)
            self.test_loss[0].append(loss)
            self.test_loss[1].append(mse)
            self.test_loss[2].append(kl)
            return loss

    def get_anim(self, name = None):
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

        anim = FuncAnimation(fig, animate, frames=len(frames), interval = 100)

        if name is None:
            return anim
        fig.suptitle(name, fontsize=14) 
          
        # saving to m4 using ffmpeg writer 
        writervideo = animation.FFMpegWriter(fps=60) 
        anim.save(name + ".mp4", writer=writervideo) 
        plt.close()

    def plot(self,datas, labels,show_result=False):
        plt.figure(1)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        for data,label in zip(datas,labels):
            durations_t = torch.tensor(data, dtype=torch.float)
            
            plt.plot(durations_t.numpy(), label = label)
            # Take 100 episode averages and plot them too
            if False and len(durations_t) >= 100:
                means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                plt.plot(means.numpy(), label = label)
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())




