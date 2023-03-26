import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

G = 6.6

class Body:
    def __init__(self, mass, position, velocity, color):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = np.zeros(2, dtype=float)
        self.color = color

        self.pos_x = []
        self.pos_y = []

    def step(self):
        self.velocity += 10*self.acceleration
        self.position += 10*self.velocity
        self.acceleration = np.zeros(2, dtype=float)

    # Compute gravity on us by a body
    def gravity(self, body):
        r = body.position - self.position
        self.acceleration += r * (G * body.mass / np.linalg.norm(r)**3)  # a_s = F/m_s = G * (m_b * m_s)/r*2 *(1 / m_s)
    

star = Body(30, np.array([0, 0], dtype=float), np.array([0, 0], dtype=float), 'orange')
planet = Body(0, np.array([0, 150], dtype=float), np.array([1.25, 0], dtype=float), 'blue')

bodies = [star, planet]

def update():
    # do steps
    for body in bodies:
        body.pos_x.append(body.position[0])
        body.pos_y.append(body.position[1])
        body.step()
    # update accelerations
    for body1 in bodies:
        for body2 in bodies:
            if body1 != body2:
                body1.gravity(body2)

T = 15_000
for t in range(T):
    update()

fig, ax = plt.subplots(1, 1, figsize = (6, 6))

def animate(end):
    start = max(0, end-100)
    ax.cla()
    for body in bodies:
        ax.plot(body.pos_x[start:end], body.pos_y[start:end], ".", color=body.color)
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)

anim = animation.FuncAnimation(fig, animate, frames = T + 1, interval = 1, blit = False)
plt.show()
