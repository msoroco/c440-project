import os
import numpy as np
import random
import argparse
import gc
from simulator import Body, Spaceship, Simulator
from animation import SimAnimation
from replay import Transition, ReplayMemory
from model import DQN
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F


def select_action(state):
    sample = random.random()
    # Exponential exploration/exploitation tradeoff
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * training_step / EPS_DECAY)
    if TEST or sample > eps_threshold:
        with torch.no_grad():
            return policy_net(torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)).argmax(1)
    else:
        return random.randint(0, n_actions-1)
    

def train():
    # Sample batch for all Transition elements (and a mask for final states)
    state_batch, action_batch, next_state_batch, reward_batch, final_state_mask, batch_size = memory.sample(BATCH_SIZE)
    state_batch = state_batch.to(device)
    action_batch = action_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    reward_batch = reward_batch.to(device)
    final_state_mask = final_state_mask.to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for next_state_batch are computed based
    # on the "older" target_net; selecting their best reward.
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[~final_state_mask] = target_net(next_state_batch).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch

    # Compute Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    # Empty GPU
    torch.cuda.empty_cache()
    gc.collect()

    # Output loss
    return loss


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Deep Q-Learning Hyperparameters')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps_start', type=float, default=0.9, help='Start value of epsilon')
    parser.add_argument('--eps_end', type=float, default=0.05, help='End value of epsilon')
    parser.add_argument('--eps_decay', type=int, default=1000, help='Epsilon decay rate')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--episodes', type=int, default=200, help='Num episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--simulation', type=str, default="sim1", help='Simulation json')
    parser.add_argument('--draw_neighbourhood', action="store_true", help='Draw neighbourhood')
    parser.add_argument('--test', action="store_true", help='Test out agent')
    parser.add_argument('--animate', action="store_true", help='Animate (whether testing or not)')
    parser.add_argument('--wandb_project', type=str, help='Save results to wandb in the specified project')
    parser.add_argument('--experiment_name', type=str, help='Name of experiment in wandb')
    parser.add_argument('--model', default='policy_net', type=str, help='Name of model to store/load')
    args, remaining = parser.parse_known_args()
    parser.add_argument('--title',  type=str, default=args.simulation, help='Title for video to save (defaults to loaded sim.json)(if --animate)')
    parser.add_argument('--save_freq',  type=int, default=args.episodes/3, help='save animation every ~ number of episodes (if --animate). Defaults to intervals of 1/3* --episodes')
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    EPS_START = args.eps_start
    EPS_END = args.eps_end
    EPS_DECAY = args.eps_decay
    TAU = args.tau
    LR = args.lr
    EPISODES = args.episodes
    MAX_STEPS = args.max_steps
    DRAW_NEIGHBOURHOOD = args.draw_neighbourhood
    TEST = args.test
    ANIMATE = args.animate

    if TEST: # There is no need to do multiple episodes when testing
        EPISODES = 1

    # setup wandb
    if args.wandb_project is not None:
        import wandb
        config = vars(args).copy()
        for k in ['draw_neighbourhood', 'test', 'animate', 'wandb_project', 'experiment_name']:
            config.pop(k, None)
        wandb.init(project=args.wandb_project, config=config, name=args.experiment_name)
        print("Initialized wandb")

    sim = Simulator(f"./simulations/{args.simulation}.json")
    sim.start()
    state_shape, n_actions = sim.info()

    print("Initialized simulator")

    policy_net = DQN(state_shape, n_actions, kernel_size=3).to(device)
    target_net = DQN(state_shape, n_actions, kernel_size=3).to(device)

    if os.path.isfile(f"./models/{args.model}.pth"):
        load_model(policy_net, f"./models/{args.model}.pth", device)

    target_net.load_state_dict(policy_net.state_dict())

    print("Initialized model")

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(5000)

    training_step = 1
    for i_episode in range(EPISODES):
        # Empty GPU
        torch.cuda.empty_cache()
        gc.collect()
        # Initialize simulation
        state = sim.start()
        # Initialize animation
        if TEST or ANIMATE:
            anim_frames = [sim.get_current_frame()]
        print("Starting episode", i_episode+1)
        # Episodic metrics
        mean_loss = 0
        total_reward = 0
        number_steps = 0
        # Run
        for t in range(MAX_STEPS):
            action = select_action(state)
            next_state, reward, terminated = sim.step(action)
            # Update episodic metrics
            total_reward += reward
            number_steps += 1

            # Store the transition in memory
            memory.push(state, action, next_state, reward, terminated)

            # Move to the next state
            state = next_state

            if TEST or ANIMATE:
                # Animate
                anim_frames.append(sim.get_current_frame())
            if not TEST:
                # Perform one step of the optimization (on the policy network)
                mean_loss += (train() - mean_loss) / (t + 1)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

            # Increment step
            training_step += 1
            if (t + 1) % 100 == 0:
                print("Step:", t+1)

            if terminated:
                print("Finished at:", t+1)
                break
        
        # Record output
        if args.wandb_project is not None:
            wandb.log({"loss": mean_loss, "reward": total_reward, "number_steps": number_steps, "episode": i_episode, "step": training_step})

        # Save model every 100 episodes
        if (i_episode + 1) % 100 == 0 and not TEST:
            save_model(policy_net, f"./models/{args.model}.pth")
        
        if TEST or ANIMATE:
            SimAnimation(sim.bodies, sim.objective, sim.limits, anim_frames, len(anim_frames), i_episode + 1, args.save_freq, args.title, 
                         DRAW_NEIGHBOURHOOD, sim.grid_radius, sim.box_width)

    if not TEST:
        save_model(policy_net, f"./models/{args.model}.pth")
