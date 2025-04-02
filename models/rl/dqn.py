from networks.network import Network
import torch
import torch.nn as nn
import torch.optim as optim

from models.rl.utils import Epsilon, TargetUpdater





class DDQN():
    def __init__(self, network_params, device = None, criterion = nn.MSELoss, 
                 gamma = 0.99, 
                 tau = 0.005, eps_params = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if not eps_params is None:
            self.eps = Epsilon(**eps_params)
        else:
            self.eps = Epsilon(eps = 0, eps_end = 0)
        
        network_params["device"] = device

        
        self.tau = tau
        self.gamma = gamma
        
        self.criterion = criterion()
        
        # Critic networks
        self.q_net1 = DQN(network_params=network_params, device = device, tau = tau)
        self.q_net2 = DQN(network_params=network_params, device = device, tau = tau)
        
        
    def train(self):
        self.q_net1.train()
        self.q_net2.train()
        self.training = True
    def eval(self):
        self.q_net1.eval()
        self.q_net2.eval()
        self.training = False

    def forward(self,x):
        return self.q_net1.q_net(x) , self.q_net2.q_net(x)
    def forward_target(self,x):
        return self.q_net1.target_net(x) , self.q_net2.target_net(x)
    
    def predict(self, x, random = None):
        
        if self.training:
            self.eps.update()

            if torch.rand(1).item() > self.eps.eps:
                with torch.no_grad():
                    action1, action2 = self.q_net1.q_net(x), self.q_net2.q_net(x)
            else:
                action1, action2 = torch.tensor([[random()]], device=self.device, dtype=torch.long), torch.tensor([[random()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                action1, action2 = self.q_net1.q_net(x) , self.q_net2.q_net(x)
        action = torch.min(action1,action2).max(1).indices.view(-1,1)
        return action1.max(1).indices.view(-1,1)
    
    def compute_target(self, non_final_next_states, action = None):

        with torch.no_grad():
            Q_target1_next = self.q_net1.target_net(non_final_next_states)
            Q_target2_next = self.q_net2.target_net(non_final_next_states)
        if action is None:
            Q_min_next = torch.min(Q_target1_next, Q_target2_next)
        else:
            Q_target1_next = Q_target1_next.gather(1,action).view(-1,1)
            Q_target2_next = Q_target2_next.gather(1,action).view(-1,1)
            Q_min_next = torch.min(Q_target1_next, Q_target2_next)
        return Q_min_next
    
    def compute_critic(self, state_batch, action_batch):
        q1 = self.q_net1.q_net(state_batch).gather(1, action_batch)
        q2 = self.q_net2.q_net(state_batch).gather(1, action_batch)
        return q1,q2

    def back_propagate(self,critic1_loss, critic2_loss):
        self.q_net1.back_propagate(critic1_loss)
        self.q_net2.back_propagate(critic2_loss)

        

    def update_target(self):
        self.q_net1.target_updater.update()
        self.q_net2.target_updater.update()
    def optimize(self, *X):
        # Compute 0.5 (Q(s, a) - (r(s,a) + gamma (pi(s+1)[Q(s+1) - alpha log(pi(s+1))])^2
        state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states = X

        q1,q2 = self.compute_critic(state_batch, action_batch)
        
        self.eval()
        action = self.predict(non_final_next_states)
        self.train() 
        Q_min_next = self.compute_target(non_final_next_states, action = action)    
        
        next_state_values = torch.zeros(len(state_batch), device=self.device).view(-1,1)
        next_state_values[non_final_mask] = Q_min_next

        Q_targets = reward_batch + (self.gamma * next_state_values)

        

        critic1_loss = 0.5 * self.criterion(q1, Q_targets)
        critic2_loss = 0.5 * self.criterion(q2, Q_targets)

        self.back_propagate(critic1_loss, critic2_loss)
        self.update_target()

    


class DQN():
    def __init__(self, network_params, device = None, criterion = nn.MSELoss, 
                 gamma = 0.99, 
                 tau = 0.005, eps_params = None, variant = "DQN"):
        self.variant = variant
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        if not eps_params is None:
            self.eps = Epsilon(**eps_params)
        else:
            self.eps = Epsilon(eps = 0, eps_end = 0)
        
        
        network_params["device"] = device

        
        self.gamma = gamma
        
        self.criterion = criterion()
        
        self.q_net = Network(**network_params)
        self.target_net = Network(**network_params)
        self.target_updater = TargetUpdater(main = self.q_net, target = self.target_net, tau = tau)   
        self.train()     

    def train(self):
        self.q_net.train()
        self.target_net.train()
        self.training = True
    def eval(self):
        self.q_net.eval()
        self.target_net.eval()
        self.training = False

    def predict(self, x, random = None):
        if self.training:
            self.eps.update()

            if torch.rand(1).item() > self.eps.eps:
                with torch.no_grad():
                    actions = self.q_net(x)
                    actions = actions.max(1).indices.view(-1,1)
            else:
                actions = torch.tensor([[random()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                actions = self.q_net(x)
                actions = actions.max(1).indices.view(-1,1)

        return actions
    
    def optimize(self, *X):
        state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states = X        
        
        state_action_values = self.q_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(len(state_batch), device=self.device)
        if self.variant == "DQN":
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            
        elif self.variant == "DoubleDQN":
            with torch.no_grad():
                next_action = self.q_net(non_final_next_states).max(1).indices.view(-1,1)
                next_q_values = self.target_net(non_final_next_states).gather(1,next_action).view(-1)   
            next_state_values[non_final_mask] = next_q_values
            

        expected_state_action_values = (next_state_values.view(-1,1) * self.gamma) + reward_batch
        loss = self.criterion(state_action_values, expected_state_action_values)
        crit_loss = loss.item()

        self.back_propagate(loss)
        
        self.target_updater.update()
    
    def back_propagate(self, loss):
        self.q_net.back_propagate(loss)