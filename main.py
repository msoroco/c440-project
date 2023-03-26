import numpy as np
import matplotlib.pyplot as plt

G = 6.6

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = np.zeros(2)

        self.pos_x = []
        self.pos_y = []

    def step(self):
        self.position += self.velocity
        self.velocity += self.acceleration
    
    # Compute gravity on us by a body
    def gravity(self, body):
        r = body.position - self.position
        self.acceleration += (r / np.linalg.norm(r)) * (G * body.mass / np.linalg.norm(r)**2)  # a_s = F/m_s = G * (m_b * m_s)/r*2 *(1 / m_s)
        print(r / np.linalg.norm(r))
    
star = Body(20, np.array([0, 0]), np.array([0, 0]))
planet = Body(0, np.array([0, 149]), np.array([5, 0]))

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

for t in range(300):
    update()

for body in bodies:
    plt.plot(body.pos_x, body.pos_y, ".")
plt.xlim(-160, 160)
plt.ylim(-160, 160)
plt.savefig("simulation.png")
plt.show()
