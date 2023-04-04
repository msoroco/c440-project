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

T = 1000


## demonstration of the simulator interface
if __name__ == '__main__':
    sim = Simulator("./sim1.json")
    
    #### testing:
    sim.start()
    states = [sim.get_current_frame()]
    
    for i in range(T):
        action = random.randint(0, 4)
        next_state, reward, terminated = sim.step(action)
        states.append(sim.get_current_frame())
        # print(sim.info())
        if terminated == True:
            break

    anim = SimAnimation(sim.bodies, states, T, True, sim.grid_radius, sim.box_width)