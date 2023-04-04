import random
import torch
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminated'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    # TODO: Give this guy device
    def sample(self, batch_size):
        """
        Returns a tensor for each of the Transition elements state, action, next_state, and reward
        and a mask tensor of final states
        """
        # In case batch_size > memory size
        batch_size = min(batch_size, len(self.memory))
        batch = random.sample(self.memory, batch_size)
        # Convert list of tuples to list of tuple elements
        state_batch, action_batch, next_state_batch, reward_batch, final_state_mask = zip(*batch)
        # Convert to tensors
        final_state_mask = torch.tensor(final_state_mask, dtype=bool)
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float)
        action_batch = torch.tensor(action_batch, dtype=int).unsqueeze(1)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float)[~final_state_mask]
        reward_batch = torch.tensor(reward_batch, dtype=torch.float).unsqueeze(1)
        return  state_batch, action_batch, next_state_batch, reward_batch, final_state_mask, batch_size

    def __len__(self):
        return len(self.memory)