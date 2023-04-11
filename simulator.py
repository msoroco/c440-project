import statistics
import numpy as np
import os
import json
import random
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

        JSON file attributes
        ---------
        * `limits`: width/height of the whole square environment. Defaults to 300.
        * `grid_radius`: The width & height in coordinates of the square frame around agent.
        * `box_width`: The width & height of a unit coordinate in the vector space.
        * `frames`: The number of past frames agent will maintain in addition to its observed frame.
        * `frame_stride`: The number of time steps between the frames the agent maintains.
        * `tolerance`: The distance (in coordinates) to objective within which agent must achieve.
          Also the distance within which agent is considered to have collided with another body.
        * `agent`:
        * `objective`: length 2 array of coordinate of the objective
        * `bodies`:
        * `start_zeros`: Have the simulation pad the missing frames with all zeros for first |`frames`| frames (default is do nothing)
        * `start_copies`: Have the simulation pad the missing frames with itself for first |`frames`| frames (`start_zeros` will default if both `True`)
        """
        json_obj = Simulator.__load_json(filepath)
        self._json_obj = json_obj
        # State information
        self.limits = json_obj["limits"]
        if self.limits is None:
            self.limits = 300
        self.grid_radius = json_obj["grid_radius"]
        self.box_width = json_obj["box_width"]
        self.frame_stride = json_obj["frame_stride"]
        self.frames = json_obj["frames"]
        self.reward_scheme = [0, self._json_obj["penalty"], self._json_obj["reward"]]
        self.tolerance = self.box_width
        # TODO: Jsonize this
        self.start_zeros = True
        self.start_copies = False

    def get_bodies_and_objective(self):
        return self.bodies, self.objective
    
    
    def info(self):
        """
        Returns the number of actions (including do nothing) and the shape of states
        """
        n_actions = len(self.agent.actions) + 1

        return self.__current_state_shape, n_actions
    

    def start(self, seed=None):
        """
        Initializes all simulation elements to starting positions

        Returns state
        """

        if seed is not None:
            random.seed(seed)
        # TODO: change this to use rng if NONE

        self.agent = Spaceship(** self._json_obj["agent"])
        self.bodies = []

        # TODO: change this to use rng if NONE
        bodies_list = self._json_obj["bodies"]
        for body in bodies_list:
            self.bodies.append(Body(**body))
        self.bodies.insert(0, self.agent)
        
        # TODO: change this to use rng if NONE
        self.objective = np.array(self._json_obj["objective"])

        # Empty past frame queue
        self.past_frames = deque([], maxlen=self.frames*self.frame_stride)
        return self.__get_state()
    

    def step(self, action : int):
        """
        Continues the simulation by one step with the given action

        Returns the next state and reward and whether objective is reached
        """
        self.agent.do_action(action)
        for body in self.bodies:
            body.step()
        for body1 in self.bodies:
            for body2 in self.bodies:
                if body1 != body2:
                    body1.gravity(body2)

        state = self.__get_state()
        terminated, reward_index = self.__get_terminated()
        reward = self.__get_reward(reward_index)
        return state, reward, terminated
    
    def __get_terminated(self):
        """
        Checkes whether the simulation should terminate if the spaceship has reached the objective,
        crashed into a planet, or went outside the simulation zone

        Returns a tuple of True if terminated or False otherwise and 0 if no termination reward, 1
        if a penalty is applied, and 2 if a reward for reach the objective
        """
        # Objective reached
        if np.linalg.norm(self.objective - self.agent.position) < self.tolerance: 
            return True, 2
        # Out of bounds
        if abs(self.agent.position[0]) > self.limits or abs(self.agent.position[1]) > self.limits:
            return True, 1
        # Check for collisions
        for body in self.bodies:
            if body != self.agent:
                # Crash
                if np.linalg.norm(body.position - self.agent.position) < self.tolerance:
                    return True, 1
        return False, 0


    def __get_reward(self, reward_index):
        # No reward if 0, penalty is 1, reward is 2
        return self.reward_scheme[reward_index] - 0.01 * (1 - self.tolerance / np.linalg.norm(self.objective - self.agent.position))
    

    def __get_state(self):
        frame = self.__get_current_frame()
        # Create state
        state = frame
        # Attach past frames
        for i in range(self.frames):
            if len(self.past_frames) >= (i+1)*self.frame_stride:
                state = np.concatenate((state, self.past_frames[i*self.frame_stride]))
            elif self.start_zeros: # TODO: If you can't attach a past frame, attach a dummy frame
                state = np.concatenate((state, np.zeros(frame.shape)))
            elif self.start_copies: # If you can't attach a past frame, attach a copy of itself
                state = np.concatenate((state, frame))
        # Update info
        self.past_frames.append(frame) # deque will automatically evict oldest frame if full
        self.__current_state_shape = state.shape
        return state
    

    def __get_current_frame(self):
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

        # Create frame       
        frame = np.stack((obstacle_grid, objective_grid), axis=0)
        return frame
    
    # for animation. May not be needed
    def get_current_frame(self):
        return self.__get_current_frame()

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
        self.velocity += 5*self.acceleration
        self.position += 5*self.velocity
        self.acceleration = np.zeros(2, dtype=float)
        self.history = np.vstack([self.history, self.position])

    # Compute gravity on us by a body
    def gravity(self, body):
        r = body.position - self.position
        self.acceleration += r * (G * body.mass / np.linalg.norm(r)**3)  # a_s = F/m_s = G * (m_b * m_s)/r*2 *(1 / m_s)


class Spaceship(Body):
    def __init__(self, position, velocity, color, speed):
        super().__init__(0, position, velocity, color)
        self.actions = [self.thrust_up, self.thrust_down, self.thrust_left, self.thrust_right]
        self.speed = speed
    
    # Accepts int for each of 4 possible actions
    def do_action(self, id):
        if id < len(self.actions):
            self.actions[id]()
        # If given id greater than 4, do nothing

    def thrust_up(self):
        self.position += np.array([0, self.speed], dtype=float)
    
    def thrust_down(self):
        self.position += np.array([0, -self.speed], dtype=float)

    def thrust_left(self):
        self.position += np.array([-self.speed, 0], dtype=float)

    def thrust_right(self):
        self.position += np.array([self.speed, 0], dtype=float)