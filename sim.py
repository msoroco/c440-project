import numpy as np

G = 6.6

class Body:
    def __init__(self, mass, position, velocity, color):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = np.zeros(2, dtype=float)
        self.color = color
        self.history = np.array([position])

    def step(self):
        self.velocity += 10*self.acceleration
        self.position += 10*self.velocity
        self.acceleration = np.zeros(2, dtype=float)
        self.history = np.vstack([self.history, self.position])

    # Compute gravity on us by a body
    def gravity(self, body):
        r = body.position - self.position
        self.acceleration += r * (G * body.mass / np.linalg.norm(r)**3)  # a_s = F/m_s = G * (m_b * m_s)/r*2 *(1 / m_s)


class Spaceship(Body):
    speed = 1

    def __init__(self, position, velocity, color):
        super().__init__(0, position, velocity, color)
        self.actions = [self.thrust_up, self.thrust_down, self.thrust_left, self.thrust_right]
    
    # Accepts int for each of 4 possible actions
    def do_action(self, id):
        self.actions[id]()

    def thrust_up(self):
        self.position += np.array([0, Spaceship.speed], dtype=float)
    
    def thrust_down(self):
        self.position += np.array([0, -Spaceship.speed], dtype=float)

    def thrust_left(self):
        self.position += np.array([-Spaceship.speed, 0], dtype=float)

    def thrust_right(self):
        self.position += np.array([Spaceship.speed, 0], dtype=float)