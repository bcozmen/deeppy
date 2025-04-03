from data.replay_buffer import Transition, ReplayBuffer
import torch


class EnvData():
    def __init__(self, env, buffer_size,  device = None):
        self.memory = ReplayBuffer(buffer_size)
        self.env = env
        self.device = device

        self.reset()
    def reset(self):
        self.state = self.env.reset()[0]
        self.state = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
    def __len__(self):
        return len(self.memory)
    def train_data(self, batch_size):
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        return state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states
    
    def collect(self, model):

        action = model.predict(self.state, self.env.action_space.sample)
        observation, reward, termination, truncation, data = self.env.step(action.item())

        ret_reward = reward

        done = (termination or truncation)

        reward = torch.tensor([reward], device=self.device, dtype = torch.float32).unsqueeze(0)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Store the transition in memory
        self.memory.push(self.state, action, next_state, reward)

        # Move to the next state
        self.state = next_state
        if done:
            self.reset()
        return done, ret_reward

