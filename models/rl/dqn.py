import torch
import torch.nn as nn


from deeppy.models.rl.rl_utils import Epsilon, TargetUpdater
from deeppy import Network
from deeppy.models import BaseModel

class DQN(BaseModel):
    dependencies = [Network, Epsilon, TargetUpdater]
    optimize_return_labels = ["Loss"]
    def __init__(self, network_params, gamma = 0.99, tau = 0.005, eps_params = {"eps":None}, variant = "DQN" ,
      device = None, criterion = nn.MSELoss(),torch_compile=False):
        super().__init__(device = device, criterion = criterion, torch_compile=torch_compile)

        self.variant = variant

        self.eps = Epsilon(**eps_params)
        self.tau = tau
        self.gamma = gamma        

        self.q_net = Network(**network_params).to(self.device)
        self.target_net = Network(**network_params).to(self.device)

        if self.torch_compile:
            self.q_net, self.target_net = torch.compile(self.q_net), torch.compile(self.target_net)
        
        self.nets = [self.q_net, self.target_net]

        self.target_updater = TargetUpdater(main = self.q_net, target = self.target_net, tau = tau)   
        

        self.params = [network_params, gamma, tau, eps_params, variant,  device]
        self.objects = [self.eps, criterion]

        self.train()

    def init_objects(self):
        self.eps, self.criterion, = self.objects

    def __call__(self, X):
        X = self.ensure(X)
        if self.training:
            self.eps.update()

            if torch.rand(1).item() > self.eps.eps:
                with torch.no_grad():
                    actions = self.q_net(X)
                    actions = actions.max(1,keepdim = True).indices
            else:
                actions = torch.tensor([self.eps.generate() for i in range(len(X))], device = self.device).unsqueeze(1)
        else:
            with torch.no_grad():
                actions = self.q_net(X)
                actions = actions.max(1,keepdim=True).indices

        return actions
    
    def optimize(self, X):
        state_batch, action_batch, next_state_batch,reward_batch, done = self.ensure(X)
        non_final_mask = torch.logical_not(done)

        state_action_values = self.q_net(state_batch).gather(1, action_batch)


        if self.variant == "DQN":
            with torch.no_grad():
                next_state_values = self.target_net(next_state_batch).max(1,keepdim=True).values * non_final_mask
            
        elif self.variant == "DoubleDQN":
            with torch.no_grad():
                next_action = self.q_net(next_state_batch).max(1,keepdim=True).indicestorch.no_grad
                next_state_values = (self.target_net(next_state_batch).gather(1,next_action) * non_final_mask)   
            

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = self.criterion(state_action_values, expected_state_action_values)
        crit_loss = loss.item()

        self.back_propagate(loss)
        self.target_updater.update()
        return loss.item()
    
    def back_propagate(self, loss):
        self.q_net.back_propagate(loss)


        

class DDQN(BaseModel):
    dependencies = [Network, Epsilon]
    optimize_return_labels = ["Mean Critic Loss"]
    def __init__(self, network_params, gamma = 0.99,tau = 0.005,  eps_params = {"eps":None}, 
     device = None, criterion = nn.MSELoss(), torch_compile = False):
        super().__init__(device = device, criterion = criterion, torch_compile=torch_compile)
        #Q = Q(s,a) -> 1
        #V = V(s)   -> dim(a)
        #Only edited by SAC in continious mode to "Q"
        self.mode = "V"
        self.eps = Epsilon(**eps_params)
        self.gamma = gamma
        
        
        # Critic networks
        self.q_net1 = DQN(network_params=network_params, device = self.device, tau = tau, torch_compile=torch_compile)
        self.q_net2 = DQN(network_params=network_params, device = self.device, tau = tau, torch_compile=torch_compile)

        self.nets = [self.q_net1, self.q_net2]
        self.params = [network_params, gamma, tau, eps_params,  device ]
        self.objects = [self.eps, criterion]

        self.train()

    def init_objects(self):
        self.eps, self.criterion, = self.objects
    
    def __call__(self, X):
        #Only for V value
        if self.mode == "Q":
            raise ValueError("Only for V networks")

        X = self.ensure(X)

        if self.training:
            self.eps.update()

            if torch.rand(1).item() > self.eps.eps:
                with torch.no_grad():
                    action = self.q_net1.q_net(X).max(1,keepdim=True).indices
            else:
                action = torch.tensor([self.eps.generate() for i in range(len(X))], device = self.device).unsqueeze(1)
        else:
            with torch.no_grad():
                action = self.q_net1.q_net(X).max(1,keepdim=True).indices
        return action

    def optimize(self, X):
        # Compute 0.5 (Q(s, a) - (r(s,a) + gamma (pi(s+1)[Q(s+1) - alpha log(pi(s+1))])^2
        #state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states = self.ensure(X)
        state_batch, action_batch, next_state_batch,reward_batch, done = self.ensure(X)
        non_final_mask = torch.logical_not(done)


        q1,q2 = self.compute_critic(state_batch, action_batch)
        
        self.eval()
        next_action = self(next_state_batch)
        self.train() 
        next_state_values = self.compute_target(next_state_batch, action = next_action) * non_final_mask 
        
        #next_state_values = torch.zeros(len(state_batch), device=self.device).view(-1,1)
        #next_state_values[non_final_mask] = Q_min_next
        Q_targets = reward_batch + (self.gamma * next_state_values)

        
        critic1_loss = 0.5 * self.criterion(q1, Q_targets)
        critic2_loss = 0.5 * self.criterion(q2, Q_targets)


        self.back_propagate(critic1_loss, critic2_loss)
        self.update_target()

        return (critic1_loss.item() + critic2_loss.item()) / 2

    #-------------------------------------------------------------------
            
    def compute_target(self, next_state_batch, action = None):
        with torch.no_grad():
            Q_target1_next = self.q_net1.target_net(next_state_batch)
            Q_target2_next = self.q_net2.target_net(next_state_batch)
        if action is None:
            Q_min_next = torch.min(Q_target1_next, Q_target2_next)
        else:
            Q_target1_next = Q_target1_next.gather(1,action)
            Q_target2_next = Q_target2_next.gather(1,action)
            Q_min_next = torch.min(Q_target1_next, Q_target2_next)
        return Q_min_next
    
    def compute_critic(self, state_batch, action_batch = None):
        if self.mode == "V":
            q1 = self.q_net1.q_net(state_batch).gather(1, action_batch)
            q2 = self.q_net2.q_net(state_batch).gather(1, action_batch)
        else:
            q1 = self.q_net1.q_net(state_batch)
            q2 = self.q_net2.q_net(state_batch)
        return q1,q2

    def back_propagate(self,critic1_loss, critic2_loss):
        self.q_net1.back_propagate(critic1_loss)
        self.q_net2.back_propagate(critic2_loss)

        

    def update_target(self):
        self.q_net1.target_updater.update()
        self.q_net2.target_updater.update()

    def forward(self,x):
        return self.q_net1.q_net(x) , self.q_net2.q_net(x)
    def forward_target(self,x):
        return self.q_net1.target_net(x) , self.q_net2.target_net(x)
    

    


