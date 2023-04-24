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
        * `limits`: 1/2 width/height of the whole square environment. Defaults to 300.
        * `grid_radius`: The width & height in coordinates of the square frame around agent.
        * `box_width`: The width & height of a unit coordinate in the vector space.
        * `frames`: The number of past frames agent will maintain in addition to its observed frame.
        * `frame_stride`: The number of time steps between the frames the agent maintains.
        * `tolerance`: The distance (in coordinates) to objective within which agent must achieve.
          Also the distance within which agent is considered to have collided with another body.
        * `agent`:
        * `random_agent_position`: wether to set the agent position to be random on each start (will overwrite given JSON position)
        * `objective`: length 2 array of coordinate of the objective
        * `bodies`:
        * `start_zeros`: Have the simulation pad the missing frames with all zeros for first |`frames`| frames (default is do nothing)
        * `start_copies`: Have the simulation pad the missing frames with itself for first |`frames`| frames (`start_zeros` will default if both `True`)
        """
        json_obj = Simulator.__load_json(filepath)
        self._json_obj = json_obj
        # State information
        self.reward_scheme = [0, self._json_obj["penalty"], self._json_obj["reward"]]
        try: self.limits = json_obj["limits"]
        except: self.limits = 300
        try: self.grid_radius = json_obj["grid_radius"]
        except: self.grid_radius = 10
        try: self.box_width = json_obj["box_width"]
        except: self.box_width = 20
        try: self.frame_stride = json_obj["frame_stride"]
        except: self.frame_stride = 1
        try: self.frames = json_obj["frames"]
        except: self.frames = 4
        try: self.penalty = self._json_obj["penalty"]
        except: self.penalty = -10
        try: self.tolerance = self._json_obj["box_width"]
        except: self.tolerance = self.box_width
        try: self.start_zeros = self._json_obj["start_zeros"]
        except: self.start_zeros = True
        try: self.start_copies = self._json_obj["self.start_copies"]
        except: self.start_copies = False
        try: self.verbose = self._json_obj["verbose"]
        except: self.verbose = False
        try: self.random_agent_position = self._json_obj["random_agent_position"]
        except: self.random_agent_position = True

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
        # TODO: change this to use rng if NONE (for velocity)
        self.agent = Spaceship(** self._json_obj["agent"])
        if self.random_agent_position is True:
            self.agent.position = np.random.uniform(-self.limits, self.limits, size=2)

        self.bodies = []
        bodies_list = self._json_obj["bodies"]
        for body in bodies_list:
            self.bodies.append(Body(**body))
        self.bodies.insert(0, self.agent)
        
        try: self.objective = np.array(self._json_obj["objective"])
        except: self.objective = np.random.uniform(-self.limits, self.limits, size=2)

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
        crashed into (or went through) a planet, or went outside the simulation zone.

        Returns a tuple of True if terminated or False otherwise and 0 if no termination reward, 1
        if a penalty is applied, and 2 if a reward for reach the objective

        Returns a tuple (terminated, reward_type).
            * `terminated`is True if terminated or False otherwise.
            * `reward_type` is:
                * 0 if no termination reward
                * 1 if a penalty is applied
                * 2 if a reward is applied for reaching the objective
        """
        terminated = False
        reward_type = 0
        # Objective reached
        if np.linalg.norm(self.objective - self.agent.position) < self.tolerance: 
            terminated = True
            reward_type = 2
            if self.verbose: print("Termination: Objective reached!")
        # Out of bounds
        elif abs(self.agent.position[0]) > self.limits or abs(self.agent.position[1]) > self.limits:
            terminated = True
            reward_type = 1
            if self.verbose: print("Termination: out of bounds")
        # Check crash or Agent going through a body:
        else:
            for body in self.bodies:
                if body != self.agent:
                    # Crash
                    if np.linalg.norm(body.position - self.agent.position) < self.tolerance:
                        terminated = True
                        reward_type = 1
                        if self.verbose: print("Termination: agent collided with a body")
                    # Agent went through a body:
                    curr_pos = self.agent.position
                    prev_pos = self.agent.history[-2]
                    agent_line = (curr_pos - prev_pos)
                    body_pos = (body.position - prev_pos)
                    projection_body = np.dot(agent_line, body_pos)/np.dot(agent_line, agent_line) * agent_line
                    # check that distance between orthogonal projection of body into agents path and body is less than tolerance
                    if np.linalg.norm(body.position - (projection_body + prev_pos)) < self.tolerance:
                        # check that body lies between the two positions of agent
                        if max(curr_pos[0], prev_pos[0]) >= (projection_body + prev_pos)[0] and min(curr_pos[0], prev_pos[0]) <= (projection_body + prev_pos)[0]:
                            if max(curr_pos[1], prev_pos[1]) >= (projection_body + prev_pos)[1] and min(curr_pos[1], prev_pos[1]) <= (projection_body + prev_pos)[1]:
                                if self.verbose: print("Termination: agent went through a body")
                                terminated = True
                                reward_type = 1
        return terminated, reward_type

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