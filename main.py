import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sim import Body, Spaceship

BOX_WIDTH = 10
GRID_R_BOXES = 10
DRAW_NEIGHBOURHOOD = True

star = Body(30, np.array([0, 0], dtype=float), np.array([0, 0], dtype=float), 'orange')
planet = Body(0, np.array([0, 150], dtype=float), np.array([1.25, 0], dtype=float), 'blue')
spaceship = Spaceship(np.array([0, 160], dtype=float), np.array([1.25, 0], dtype=float), 'black')
objective = np.array([-100, -100], dtype=float)

bodies = [star, planet, spaceship]

# grid_r_boxes = number of boxes from the center
# TODO: add new channels for past frames
def get_state(agent, objective, bodies, grid_r_boxes, box_width, frames=4, step_size=1):
    grid_radius = (grid_r_boxes + 0.5) * box_width
    obstacle_grid = np.zeros((2*grid_r_boxes+1, 2*grid_r_boxes+1))
    objective_grid = np.zeros((2*grid_r_boxes+1, 2*grid_r_boxes+1))
    # assign objective
    position = objective - agent.position
    position = np.sign(position)*np.minimum(grid_radius*np.ones(2), np.abs(position))
    # TODO: this all kind of depends how you want the grids to look (it's just a bunch of horizontal flips or whatever)
    index = np.floor((grid_radius-1 - position)/box_width).astype(int)
    objective_grid[index[1], 2*grid_r_boxes - index[0]] = 1
    # TODO: python-ize this (probably easy)
    for body in bodies:
        if body != agent:
            position = body.position - agent.position
            # need to do transform for rotations
            if np.linalg.norm(position, ord=1) <= grid_radius:
                index = np.floor((grid_radius-1 - position)/box_width).astype(int)
                obstacle_grid[index[1], 2*grid_r_boxes - index[0]] = 1
    return np.stack((obstacle_grid, objective_grid), axis=0)           

## example 
# grid = get_state(spaceship, objective, bodies, GRID_R_BOXES, BOX_WIDTH)
# print(grid)

T = 15000
for t in range(T):
    spaceship.do_action(3)
    # do steps
    for body in bodies:
        body.step()
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
            rect = drawShipneighbourhood(body.history[end])
            ax.add_patch(rect)
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)

def drawShipneighbourhood(position):
    radius = (GRID_R_BOXES + 0.5) * BOX_WIDTH
    rectangle = plt.Rectangle((position[0] - radius, position[1] - radius), 2*radius, 2*radius, fill=False)
    return rectangle


anim = animation.FuncAnimation(fig, animate, frames = T + 1, interval = 1, blit = False)
plt.show()
