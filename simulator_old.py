import statistics
import numpy as np
import os
import json
from collections import deque

G = 6.6

class Simulator:
    def __init__(self, filepath : str):
        """
        Create a simulation from a JSON

        Parameters
        ----------
        filepath : str
            Path to json file
        """
        json_obj = Simulator.__load_json(filepath)
        self.agent = None
        self.bodies = None
        self.objective = None
        # State information
        self.grid_radius = json_obj["grid_radius"]
        self.box_width = json_obj["box_width"]
        self.frame_stride = json_obj["frame_stride"]
        self.frames = json_obj["frames"]
        self.past_frames = deque([], maxlen=self.frames*self.frame_stride) # To avoid recomputation
        pass


    def info(self):
        """
        Returns the number of actions (including do nothing) and the shape of states
        """
        return None, None
    

    def start(self):
        """
        Initializes all simulation elements to starting positions

        Returns state
        """
        return None
    

    def step(self, action : int):
        """
        Continues the simulation by one step with the given action

        Returns the next state and reward
        """
        # spaceship.do_action(3)
        # for body in bodies:
        #     body.step()
        # statistics.append(get_state(spaceship, objective, bodies, GRID_RADIUS, BOX_WIDTH))
        # for body1 in bodies:
        #     for body2 in bodies:
        #         if body1 != body2:
        #             body1.gravity(body2)
        return None, None
    

    def __get_reward(self):
        return None
    

    def __get_state(self):
        radius = (self.grid_radius + 0.5) * self.box_width
        obstacle_grid = np.zeros((2*self.grid_radius+1, 2*self.grid_radius+1))
        objective_grid = np.zeros((2*self.grid_radius+1, 2*self.grid_radius+1))

        # Assign objective
        # Transform and shift to bottom left corner of grid
        position = (self.objective - self.agent.position)
        # TODO: rotation goes here
        position = position + radius*np.ones(2)
        index = np.clip(np.floor(position/self.box_width).astype(int), 0, 2*self.grid_radius)
        objective_grid[index[1], index[0]] = 1

        # Assign obstacles
        for body in self.bodies:
            if body != self.agent:
                position = body.position - self.agent.position
                # TODO: rotation goes here
                if np.max(np.abs(position)) <= radius:
                    position = position + radius*np.ones(2)
                    index = np.clip(np.floor(position/self.box_width).astype(int), 0, 2*self.grid_radius)
                    obstacle_grid[index[1], index[0]] = 1

        # Create state       
        frame = np.stack((obstacle_grid, objective_grid), axis=0)
        state = frame

        # Attach past states
        for i in range(self.frames):
            if len(self.past_frames) >= (i+1)*self.frame_stride:
                state = np.concatenate((state, self.past_frames[(i+1)*self.frame_stride - 1]))
        self.past_frames.append(frame)
        return state
    

    def __load_json(filepath):
        with open(filepath) as json_file:
            json_obj = json.load(json_file)
        return json_obj


class Body:
    def __init__(self, mass, position, velocity, color):
        self.mass = float(mass)
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
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
        if id < len(self.actions):
            self.actions[id]()
        # If given id greater than 4, do nothing

    def thrust_up(self):
        self.position += np.array([0, Spaceship.speed], dtype=float)
    
    def thrust_down(self):
        self.position += np.array([0, -Spaceship.speed], dtype=float)

    def thrust_left(self):
        self.position += np.array([-Spaceship.speed, 0], dtype=float)

    def thrust_right(self):
        self.position += np.array([Spaceship.speed, 0], dtype=float)