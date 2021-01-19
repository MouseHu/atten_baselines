import gym
import numpy as np
from gym.spaces import Box


class JumperEnv(gym.Env):
    def __init__(self, horizon=1000, noise_range=0.1):
        self.horizon = horizon
        # self.max_step = horizon
        self.noise_range = noise_range
        self.num_steps = 0
        self.state = 0
        self.observation_space = Box(np.array([0, 0]), np.array([1, horizon]))
        self.action_space = Box(np.array([-1]), np.array([1]))

    def reset(self):
        self.state = np.random.rand()
        self.num_steps = 0
        return np.array([self.state,0])

    def step(self, action):
        self.state = np.clip(self.state + action, 0, 1)
        reward = float(self.state > .5) - float(action) ** 2
        self.num_steps += 1
        state = np.array([self.state, self.num_steps])
        done = self.num_steps >= self.horizon
        info = dict()

        return state, reward, done, info
