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
from blackscholes import geometric_brownian_motion, CallOption


class EuropeanOptionEnv(gym.Env):
    def __init__(self, t=1000, n=10, s0=49, k=50, max_stock=1, vola=.02):
        self.T = t
        # self.option = CallOption(t, k, n)

        self.dt = 1. / self.T
        self.max_stock = max_stock
        self.k = 1
        self.vola = vola
        self.s0 = s0 / k
        self.number_options = n
        self.option = CallOption(t, self.k, 1)

        # self.action_space = spaces.Discrete(2 * max_stock + 1)
        self.action_space = spaces.Box(low=np.array([0.]), high=np.array([2 * max_stock]))
        # (S, tau, stocks)
        self.observation_space = spaces.Box(low=np.array([0., 0., -self.max_stock]),
                                            high=np.array([1., 10., self.max_stock]))
        # self.observation_space = spaces.Tuple((
        #     spaces.Box(low=np.array([0.]), high=np.array([10.])),   # S
        #     spaces.Box(low=np.array([0.]), high=np.array([1.])),    # tau
        #     spaces.Box(low=np.array([-self.max_stock]),
        #                high=np.array([self.max_stock]))  # stocks
        # ))

        self.time = 0
        self.stock = 0
        self.underlying = None
        self.cash = 0
        self._reset()

    def _convert_action_to_stock(self, action):
        return action - self.max_stock

    def _step(self, action):
        assert self.action_space.contains(action)

        stock_to_buy = action - self.max_stock
        s = self.underlying[self.time]
        self.stock += stock_to_buy
        self.cash -= s * stock_to_buy

        self.time += 1

        done = True if self.T and self.time >= self.T else False

        reward = 0.
        if done:
            call = self.option.calc(self.option.expiry, self.vola, s)
            reward = - np.abs(self.cash + s * self.stock + np.clip(s - self.k, 0., np.infty)) #call['npv']
            return np.array([s, 0, self.stock]), reward, done, {}

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        underlying = self.underlying[self.time]
        tau = (self.T - self.time) * 1.0 / self.T
        number_stocks = self.stock
        return np.array([underlying, tau, number_stocks])

    def _reset(self):
        self.time = 0
        self.stock = 0
        self.cash = 0
        t, self.underlying = geometric_brownian_motion(t=1., dt=self.dt, s0=self.s0/self.k)
        self.underlying = self.underlying / self.k
        # print(len(self.underlying), self.dt)
        return self._get_obs()
