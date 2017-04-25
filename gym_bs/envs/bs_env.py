"""
Environment with an european option.

Hedging optimal policy is the Black Scholes delta hedging.

This environment has just two actions.
Action 0 sells.
Action 1 buys

Optimal policy: black scholes delta

Optimal value function: v(0)=1 (there is only one state, state 0)
"""

import gym
from gym import spaces
import numpy as np
from blackscholes import geometric_brownian_motion


class EuropeanOptionEnv(gym.Env):
    def __init__(self):
        self.T = 1000
        self.dt = 1. / self.T
        self.time = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=np.array([0.]), high=np.array([10.])),   # S
            spaces.Box(low=np.array([0.]), high=np.array([1.])),    # tau
            spaces.Box(low=np.array([-100]), high=np.array([100]))  # stocks
        ))
        self._reset()

    def _step(self, action):
        assert self.action_space.contains(action)

        self.time += 1

        done = True if self.T and self.time >= self.T else False

        # reward = np.clip(K - S, 0, 100)

        reward = 0.0
        if done:
            reward = 1.0

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        pass
        return 0 #self.observation_space

    def _reset(self):
        self.time = 0
        self.underlying = geometric_brownian_motion(t=1., dt=self.dt)

        return self._get_obs()
