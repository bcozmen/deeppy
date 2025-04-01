import random
from collections import deque, namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer():
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer_object = Transition

        self.buffer = deque(maxlen=capacity)
        self.illegal_count = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(Transition(*args))
    def add(self, items):
        if not type(items) is list:
            items = [items]
        if not all(isinstance(element, self.buffer_object) for element in items):
            raise ValueError('You have tried to add an invalid objects to the buffer')
        self.buffer.extend(items)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)




