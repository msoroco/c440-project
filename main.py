import numpy as np
import random
import argparse
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
    parser = argparse.ArgumentParser(description='Deep Q-Learning Hyperparameters')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps_start', type=float, default=0.9, help='Start value of epsilon')
    parser.add_argument('--eps_end', type=float, default=0.05, help='End value of epsilon')
    parser.add_argument('--eps_decay', type=int, default=1000, help='Epsilon decay rate')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum steps per episode')

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    EPS_START = args.eps_start
    EPS_END = args.eps_end
    EPS_DECAY = args.eps_decay
    TAU = args.tau
    LR = args.lr
    MAX_STEPS = args.max_steps

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
            next_state, reward, terminated = sim.step(action) # TODO: return None for final states (ms: i added terminated instead)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

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
