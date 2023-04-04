import numpy as np
import random
from simulator import Body, Spaceship, Simulator
from animation import SimAnimation
from replay import Transition, ReplayMemory
from model import DQN
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F

BOX_WIDTH = 20
GRID_RADIUS = 8
DRAW_NEIGHBOURHOOD = True


def select_action(state, time_step):
    sample = random.random()
    # exponential exploration/exploitation tradeoff
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * time_step / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(1)
    else:
        return random.randint(0, n_actions-1)
    

def train():
    # Sample batch for all Transition elements (and a mask for final states)
    state_batch, action_batch, next_state_batch, reward_batch, final_state_mask = memory.sample(BATCH_SIZE).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for next_state_batch are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[~final_state_mask] = target_net(next_state_batch).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: make this command line args
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    MAX_STEPS = 10000

    sim = Simulator("./sim1.json")
    state_shape, n_actions = sim.info()

    policy_net = DQN(state_shape, n_actions).to(device)
    target_net = DQN(state_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(10000)

    for i_episode in range(1):
        # Initialize simulation
        state = sim.start()
        for t in MAX_STEPS:
            action = select_action(state)
            next_state, reward, terminated = sim.step(action)

            # Store the transition in memory
            memory.push(state, action, next_state, reward, terminated)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            train()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            # if next_state is None:
            #     break
            if terminated == True:
                break
