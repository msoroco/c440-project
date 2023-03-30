import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sim import Body, Spaceship

BOX_WIDTH = 20
GRID_RADIUS = 4
DRAW_NEIGHBOURHOOD = True

star = Body(30, np.array([0, 0], dtype=float), np.array([0, 0], dtype=float), 'orange')
planet = Body(0, np.array([0, 150], dtype=float), np.array([1.25, 0], dtype=float), 'blue')
spaceship = Spaceship(np.array([0, 160], dtype=float), np.array([1.25, 0], dtype=float), 'black')
objective = np.array([-100, -100], dtype=float)

bodies = [star, planet, spaceship]
states = []

# grid_radius = number of boxes from the center
# TODO: add new channels for past frames
def get_state(agent, objective, bodies, grid_radius, box_width, frames=4, step_size=1):
    radius = (grid_radius + 0.5) * box_width
    obstacle_grid = np.zeros((2*grid_radius+1, 2*grid_radius+1))
    objective_grid = np.zeros((2*grid_radius+1, 2*grid_radius+1))
    # assign objective
    # transform and shift to bottom left corner
    position = (objective - agent.position) + radius*np.ones(2)
    # position = np.sign(position)*np.minimum(radius*np.ones(2), np.abs(position))
    # TODO: this all kind of depends how you want the grids to look (it's just a bunch of horizontal flips or whatever)
    index = np.clip(np.floor(position/box_width).astype(int), 0, 2*grid_radius)
    objective_grid[index[0], index[1]] = 1
    # TODO: python-ize this (probably easy)
    for body in bodies:
        if body != agent:
            position = body.position - agent.position
            # need to do transform for rotations
            if np.max(np.abs(position)) <= radius:
                position = position + radius*np.ones(2)
                index = np.clip(np.floor(position/box_width).astype(int), 0, 2*grid_radius)
                obstacle_grid[index[0], index[1]] = 1
    return np.stack((obstacle_grid, objective_grid), axis=0)           

## example 
# grid = get_state(spaceship, objective, bodies, GRID_RADIUS, BOX_WIDTH)
# print(grid)

T = 1000
for t in range(T):
    spaceship.do_action(3)
    # do steps
    for body in bodies:
        body.step()
    # add state to states
    states.append(get_state(spaceship, objective, bodies, GRID_RADIUS, BOX_WIDTH))
    # update accelerations
    for body1 in bodies:
        for body2 in bodies:
            if body1 != body2:
                body1.gravity(body2)

fig, ax = plt.subplots(1, 1, figsize = (6, 6))

def animate(end):
    start = max(0, end-100)
    ax.cla()
    for body in bodies:
        ax.plot(body.history[start:end, 0], body.history[start:end, 1], ".", color=body.color)
        if isinstance(body, Spaceship) and DRAW_NEIGHBOURHOOD:
            draw_state(states[end-1], body.history[end-1], ax)
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)


def draw_state(state, position, ax):
    position = position - np.ones(2) * (GRID_RADIUS + 0.5) * BOX_WIDTH
    for i in range(0, 2*GRID_RADIUS+1):
        for j in range(0, 2*GRID_RADIUS+1):
            if state[0, i, j] == 1:
                ax.add_patch(plt.Rectangle((position[0] + i*BOX_WIDTH, position[1] + j*BOX_WIDTH), BOX_WIDTH, BOX_WIDTH, fill=True))
            ax.add_patch(plt.Rectangle((position[0] + i*BOX_WIDTH, position[1] + j*BOX_WIDTH), BOX_WIDTH, BOX_WIDTH, fill=False))


anim = animation.FuncAnimation(fig, animate, frames = T + 1, interval = 1)
plt.show()
