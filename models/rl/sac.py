from networks.FFN import FFN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import copy

from models.rl.dqn import DDQN



class Discrete_SAC():
    def __init__(self, ddqn_params, pnet_params, alpha_lr, gamma =0.99, device = None, criterion = nn.MSELoss, target_entropy_coeff = 1.):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.gamma = gamma
            
        pnet_params["device"] = device
        ddqn_params["device"] = device

        self.criterion = criterion()


        # Alpha
        self.target_entropy = target_entropy_coeff * - torch.log(torch.tensor(pnet_params["layers"][-1]))
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.AdamW(params=[self.log_alpha], lr=alpha_lr, amsgrad = False)

        # Actor
        self.policy_net = FFN(**pnet_params)

        # Critic networks
        self.ddqn = DDQN(**ddqn_params)
        self.train()

    def train(self):
        self.ddqn.train()
        self.policy_net.train()
        self.training = True
    def eval(self):
        self.ddqn.eval()
        self.policy_net.eval()
        self.training = False

    def predict(self, x, random = None):
        with torch.no_grad():
            action, action_probs, log_pis = self.get_action(x)
        entropy = -1*(action_probs * log_pis).detach().cpu().numpy().sum(1)
        return action

    def optimize(self, *X):
        states, actions, rewards, non_final_mask, non_final_next_states = X

        # update critic
        c1_loss, c2_loss = self.critic_loss(states, actions, rewards, non_final_mask, non_final_next_states)

        # Update Actor
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, action_probs, log_pis = self.policy_loss(states, current_alpha.to(self.device))

        # Alpha loss
        alpha_loss = self.alpha_loss(action_probs.detach().cpu(), log_pis.detach().cpu())
        self.alpha = self.log_alpha.exp().detach()

        # Soft target update
        self.ddqn.update_target()

    def get_action(self, x):
        action_probs = self.policy_net(x)

        if not self.training:
            action = action_probs.max(1).indices.view(-1,1)
        else:
            dist = Categorical(action_probs)

            action = dist.sample().view(-1,1)

        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs+z)

        return action, action_probs, log_action_probs
    
    def critic_loss(self, states, actions, rewards, non_final_mask, non_final_next_states):
        # Compute 0.5 (Q(s, a) - (r(s,a) + gamma (pi(s+1)[Q(s+1) - alpha log(pi(s+1))])^2
        with torch.no_grad():
            _, action_probs, log_pis = self.get_action(non_final_next_states)
        Q_min_next = self.ddqn.compute_target(non_final_next_states)
        


        Q_target_next = (action_probs * (Q_min_next - self.alpha.to(self.device) * log_pis)).sum(1)

        next_state_values = torch.zeros(len(states), device=self.device)
        next_state_values[non_final_mask] = Q_target_next

        Q_targets = rewards + (self.gamma * next_state_values.unsqueeze(-1))

        # compute critic loss
        q1,q2 = self.ddqn.compute_critic(states,actions)
        

        critic1_loss = 0.5 * self.criterion(q1, Q_targets)
        critic2_loss = 0.5 * self.criterion(q2, Q_targets)

        self.ddqn.back_propagate(critic1_loss,critic2_loss)

        return critic1_loss, critic2_loss        

    def alpha_loss(self, action_probs, log_pis):
        # Computes pi(s).T[-alpha(log(pi(s)) + H)] and backpropagate
        alpha_loss = (action_probs * (-self.log_alpha.exp() * (log_pis + self.target_entropy))).sum(1).mean()
        self.back_propagate(self.alpha_optimizer, alpha_loss)
        return alpha_loss

    def policy_loss(self, states, alpha):
        # Compute pi(s).T [alpha* log(pi(s)) - Q(s)]

        # Get pi and log(pi)
        _, action_probs, log_pis = self.get_action(states)

        # Get Q
        with torch.no_grad():
            min_Q = torch.min(*self.ddqn.forward(states))

        # Compute and back propagate
        actor_loss = (action_probs * ((alpha * log_pis) - min_Q)).sum(1).mean()
        self.back_propagate(self.policy_net.optimizer, actor_loss)
        return actor_loss, action_probs, log_pis




    def back_propagate(self, optim, loss):
        optim.zero_grad()
        loss.backward()
        optim.step()
