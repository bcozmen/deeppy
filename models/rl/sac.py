
import torch
import torch.nn as nn
import torch.optim as optim
import copy


from deeppy import Network
from deeppy.models import BaseModel

from deeppy.models.rl.dqn import DDQN
from torch.cuda.amp import GradScaler


class SAC(BaseModel):
    """
    Discrete   - https://arxiv.org/abs/1910.07207
    Continuous - https://arxiv.org/abs/1812.05905
    """
    dependencies = [Network, DDQN]
    optimize_return_labels = ["Mean Critic Loss", "Actor Loss", "Alpha Loss"]
    def __init__(self, ddqn_params, pnet_params, alpha_lr, gamma =0.99, target_entropy = -1. , 
        device = None, criterion = nn.MSELoss(), amp=False,
        mode = "discrete", continuous_action = torch.tanh):
        super().__init__(device = device, criterion = criterion, amp=amp)

        self.mode = mode
        self.continuous_action = continuous_action

        self.gamma = gamma

        # Alpha
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor([0.0], requires_grad=True, dtype = torch.float32, device = self.device)
        self.alpha = self.log_alpha.exp().detach()
        
        self.alpha_optimizer = optim.AdamW(params=[self.log_alpha], lr=alpha_lr, amsgrad = False)

        # Actor
        self.policy_net = Network(**pnet_params).to(self.device)



        # Critic networks
        self.ddqn = DDQN(**ddqn_params, device = device)
        if self.mode != "discrete":
            self.ddqn.mode = "Q"

        self.params = [ddqn_params, pnet_params, alpha_lr, gamma, self.target_entropy, device]
        self.nets = [self.ddqn, self.policy_net]
        self.objects = [criterion]

        if self.amp:
            print("AMP is not supported with this model")
            self.amp=False
            self.scaler = GradScaler(enabled=self.use_amp)
            self.optimizers = []




    def __call__(self,X):
        return self.predict(X)

    def predict(self, X):
        with torch.no_grad():
            action, action_probs, log_pis = self.get_action(X)
        return action


    def get_action(self, X):
        action_probs = self.policy_net(X) #(Batch, 2dim)

        if self.mode == "discrete":
            if not self.training:
                action = action_probs.max(1,keepdim=True).indices
            else:
                action = torch.multinomial(action_probs, num_samples = 1)



            z = action_probs == 0.0
            z = z.float() * 1e-8
            log_action_probs = torch.log(action_probs+z)
        else:
            len_latent = action_probs.shape[1]//2
            mu, std = action_probs[:, :len_latent], torch.abs(action_probs[:, len_latent:])
            std = torch.clamp(std, min = 1e-6, max = 4)


            normal = torch.distributions.Normal(mu, std)
            x_t = normal.rsample()
            log_action_probs = normal.log_prob(x_t)
            if not self.training:
                x_t = mu

            action = self.continuous_action(x_t) #(Batch, 1)
            log_action_probs -= torch.log((3 * (1 - (action)).pow(2)) + 1e-6)
            

            #log_action_probs -= (2*( - x_t - nn.functional.softplus(-2*x_t)) + torch.log(torch.tensor(2)) ).sum(1)
            

        return action, action_probs, log_action_probs

    def optimize(self, X):
        state_batch, action_batch, next_state_batch,reward_batch, done = X
        """
        print(state_batch.shape)
        print(action_batch.shape)
        print(next_state_batch.shape)
        print(reward_batch.shape)
        print(done.shape)
        """    
        non_final_mask = torch.logical_not(done)

        # update critic
        c1_loss, c2_loss = self.critic_loss(state_batch, action_batch, reward_batch, non_final_mask, next_state_batch)

        # Update Actor
        actor_loss, action_probs, log_pis = self.policy_loss(state_batch, self.alpha)

        # Alpha loss
        alpha_loss = self.alpha_loss(action_probs.detach(), log_pis.detach())
        

        # Soft target update
        self.ddqn.update_target()

        return (c1_loss.item() + c2_loss.item())/2, actor_loss.item(), alpha_loss.item()

    def critic_loss(self, states, actions, rewards, non_final_mask, next_states):
        # Compute 0.5 (Q(s, a) - (r(s,a) + gamma (pi(s+1)[Q(s+1) - alpha log(pi(s+1))])^2
        with torch.no_grad():
            next_action, next_action_probs, log_pis = self.get_action(next_states)

        
        if self.mode == "discrete":
            q1,q2 = self.ddqn.compute_critic(states,actions)
            Q_min_next = self.ddqn.compute_target(next_states)
            next_state_values = (next_action_probs * (Q_min_next - self.alpha * log_pis)).sum(1).unsqueeze(-1) * non_final_mask
        else:
            state_actions = torch.cat((states,actions), dim = 1).to(self.device, non_blocking = True)
            next_state_action = torch.cat((next_states, next_action), dim=1).to(self.device, non_blocking = True)
            
            q1,q2 = self.ddqn.compute_critic(state_actions) #(Batch,1)

            Q_min_next = self.ddqn.compute_target(next_state_action) #(Batch,1)
            next_state_values = (Q_min_next - self.alpha * log_pis) * non_final_mask

        
        Q_targets = rewards + (self.gamma * next_state_values)

        

        critic1_loss = 0.5 * self.criterion(q1, Q_targets)
        critic2_loss = 0.5 * self.criterion(q2, Q_targets)

        self.ddqn.back_propagate(critic1_loss,critic2_loss)


        return critic1_loss, critic2_loss        

    def alpha_loss(self, action_probs, log_pis):
        # Computes pi(s).T[-alpha(log(pi(s)) + H)] and backpropagate
        if self.mode == "discrete":
            alpha_loss = (action_probs * (-self.log_alpha.exp() * (log_pis + self.target_entropy))).sum(1).mean()
        else:
            alpha_loss = (-self.log_alpha.exp() * (log_pis + self.target_entropy)).mean()
        self.back_propagate(self.alpha_optimizer, alpha_loss)
        self.alpha = self.log_alpha.exp().detach()
        return alpha_loss

    def policy_loss(self, states, alpha):
        # Compute pi(s).T [alpha* log(pi(s)) - Q(s)]

        # Get pi and log(pi)
        actions, action_probs, log_pis = self.get_action(states)

        # Get Q
        #with torch.no_grad():
        if self.mode == "discrete":
            min_Q = torch.min(*self.ddqn.forward(states))
        else:
            state_actions = torch.cat((states,actions), dim = 1)
            min_Q = torch.min(*self.ddqn.forward(state_actions))
        
        # Compute and back propagate
        if self.mode == "discrete":
            actor_loss = (action_probs * ((alpha * log_pis) - min_Q)).sum(1).mean()
        else:
            actor_loss = (alpha * log_pis - min_Q).mean()


        self.policy_net.back_propagate(actor_loss)
        return actor_loss, action_probs, log_pis


    def back_propagate(self, optim, loss):
        optim.zero_grad()
        loss.backward()
        optim.step()
