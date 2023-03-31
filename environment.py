import numpy as np
import random


class environment:

    # box_width: width of one coordinate relative to the vector space
    # grid_radius: half the width & height of the agents viewable square area
    # n_bodies: number of bodies to plot
    # tolerance: How close to goal agent must reach to be considered done. Defaults to box_width
    def __init__(self, environment_width, box_width, grid_radius, bodies=['star'], animate=True, tolerance=None, agent_colour='black', body_colours=None) -> None:
        self.box_width = box_width
        self.grid_radius = grid_radius
        self.n_bodies = len(bodies)
        self.bodies = []
        self.bodies_list = bodies
        self.body_colours = body_colours
        self.animate = animate
        self.tolerance = self.box_width

        # self.bodies = []
        # self.agent = 
        self.agent_colour = agent_colour
        self.star_mass = 30
        self.planet_mass = 0.00009

        pass

    # compute the environment state after applying the given action.
    # returns a 4-tuple: (observation, reward, done, info)
    #   observation: the last 4 frames the agent observed.
    #           frame: 2-tuple: (objective, bodies) grid
    #   reward: inversely proportional to the distance between agent and goal
    #   done:   boolean. Did agent reach goal within tolerance?
    #   info:   any other information needed for debugging
    def step(self, action):
        pass
        

    # (re)set the environment.
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        # TODO: change this to use rng
        position = np.array([0, 160], dtype=float)
        velocity = np.array([1.25, 0], dtype=float)
        self.agent = Spaceship(position, velocity, self.agent_colour)

        # TODO: change this to use rng
        for body in self.bodies_list:
            if self.bodies_list[body] == 'star':
                self.bodies[body] = Body(self.star_mass, np.array([0, 0], dtype=float), np.array([0, 0], dtype=float), 'orange')
            if self.bodies_list[body] == 'planet':
                self.bodies[body] = Body(self.planet_mass, np.array([0, 150], dtype=float), np.array([1.25, 0], dtype=float), 'blue')
        self.bodies.insert(0, self.agent)
        
        # TODO: change this to use rng
        self.objective = np.array([-100, -100], dtype=float)
        
        return
    
    # TODO: if necessary
    def _get_info(self):
        return
    
    # return the most recent frame.
    #   frame : 2-tuple: (objective, bodies) grid
    def _get_state(self):
        return

    # return the agent's current last 4 frames
    #   frame : 2-tuple: (objective, bodies) grid
    def _get_observation(self):
        return
    



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