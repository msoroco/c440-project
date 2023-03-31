import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_shape : list, n_actions : int, n_convs=2, kernel_size=5, pool_size=2, n_out_channels=16, n_lins=3):
        """
        Parameters
        ----------
        state_shape : list of int
            The shape of an input state as (channels, height, width)
        n_actions : int
            The number of actions
        n_convs : int, optional
            The number of Conv2d layers
        kernel_size : int, optional
            The kernel size of the Conv2d layers
        pool_size : int, optional
            The size of the pooling layers
        n_out_channels: int, optional
            The number of channels at the end of the convolutional layers
        n_lins: int, optional
            The number of linear layers
        """
        super(DQN, self).__init__()
        state_shape = np.array(state_shape)

        # conv layer setup
        channels = np.linspace(state_shape[0], n_out_channels, num=n_convs+1).astype(int)
        self.convs = nn.ModuleList()
        for i in range(n_convs):
            self.convs.append(nn.Conv2d(channels[i], channels[i+1], kernel_size))
            if i < n_convs-1: 
                self.convs.append(nn.MaxPool2d(pool_size))
                state_shape[1:] = state_shape[1:] - (pool_size - 1)
            state_shape[1:] = state_shape[1:] - (kernel_size - 1)
        state_shape[0] = channels[n_convs]

        # lin layer setup
        lin_sizes = np.linspace(np.prod(state_shape), n_actions, num=n_lins+1).astype(int)
        self.lins = nn.ModuleList()
        for i in range(n_lins):
            self.lins.append(nn.Linear(lin_sizes[i], lin_sizes[i+1]))
            if i < n_lins-1: 
                self.lins.append(nn.ReLU())

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.lins(x)
        return x