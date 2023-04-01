import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from simulator import Spaceship

class SimAnimation():
    def __init__(self, bodies, states, iterations, draw_neighbourhood=False, grid_radius=None, box_width=None):
        self.draw_neighbourhood = draw_neighbourhood
        if self.draw_neighbourhood and (grid_radius==None or box_width==None):
            raise Exception("If draw_neighbourhood is set to True, grid_radius and box_width must be specified") 
        self.grid_radius = grid_radius
        self.box_width = box_width
        self.bodies = bodies
        self.states = states
        fig, self.ax = plt.subplots(1, 1, figsize = (6, 6))
        anim = animation.FuncAnimation(fig, self.animate, frames = iterations+1, interval = 1)
        plt.show()

    def animate(self, end):
        start = max(0, end-100)
        self.ax.cla()
        for body in self.bodies:
            self.ax.plot(body.history[start:end+1, 0], body.history[start:end+1, 1], ".", color=body.color)
            if isinstance(body, Spaceship) and self.draw_neighbourhood:
                self.draw_state(self.states[end], body.history[end])
        self.ax.set_xlim(-300, 300)
        self.ax.set_ylim(-300, 300)


    def draw_state(self, state, position):
        position = position - np.ones(2) * (self.grid_radius + 0.5) * self.box_width
        for i in range(0, 2*self.grid_radius+1):
            for j in range(0, 2*self.grid_radius+1):
                if state[0, i, j] == 1:
                    self.ax.add_patch(plt.Rectangle((position[1] + j*self.box_width), position[0] + i*self.box_width,
                                                    self.box_width, self.box_width, fill=True))
                self.ax.add_patch(plt.Rectangle((position[1] + j*self.box_width), position[0] + i*self.box_width, 
                                                self.box_width, self.box_width, fill=False))