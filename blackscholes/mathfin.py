import numpy as np
from scipy.stats import norm


def geometric_brownian_motion(mu=0., sigma=1., t=2., dt=0.01, s0=1.):
    """
    Generate geometric brownian motion.
    :param mu: 
    :param sigma: 
    :param t: 
    :param dt: 
    :param s0: 
    :return: 
    """
    N = int(round(t / dt))
    tt = np.linspace(0, t, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  ### standard brownian motion ###
    X = (mu - 0.5 * sigma ** 2) * tt + sigma * W
    S = s0 * np.exp(X)  ### geometric brownian motion ###
    return tt, S


def blackScholes(tau, S, K, sigma):
    """
    Computes a set of theoretical quantities associated to a call option in the Black Scholes model.
    :param tau: time to expiry 
    :param S: the underlying's value
    :param K: the strike of the option
    :param sigma: the implied volatility
    :return: dict containing
        npv: net present value of the option
        delta: the sensitivity to the change to the underlying
        gamma: the sensitivity to the change in delta
        vega: the sensitivity to the change in volatility
        theta: the sensitivity to the passage of time
    """
    sqrt_tau = np.sqrt(tau)
    sigma_sqrt_tau = sigma * sqrt_tau
    d1 = np.log(S / K) / sigma / sqrt_tau + 0.5 * sigma_sqrt_tau
    d2 = d1 - sigma_sqrt_tau
    npv = (S * norm.cdf(d1) - K * norm.cdf(d2))
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma_sqrt_tau)
    vega = S * norm.pdf(d1) * sqrt_tau
    theta = -.5 * S * norm.pdf(d1) * sigma / sqrt_tau
    return {'npv': npv, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}


class CallOption(object):
    """
    Represents a call option in the Black Scholes framework
    """
    def __init__(self, t, k, n):
        """

        :type t: int
        :type k: int
        :type n: int
        """
        self.expiry = t
        self.strike = k
        # self.start = start  # day to sell option
        self.N = n

    def calc(self, today, vola, underlying):
        # if today < self.start:
        #     return {'delta': 0, 'npv': 0, 'vega': 0, 'gamma': 0, 'theta': 0, 'intrinsic': 0}
        # if today > self.expiry:
        #     return {'delta': 0, 'npv': 0, 'vega': 0, 'gamma': 0, 'theta': 0, 'intrinsic': 0}
        # if today == self.expiry:
        #     return {'delta': 0, 'npv': 0, 'vega': 0, 'gamma': 0, 'theta': 0, 'intrinsic': self.N * max(0, underlying - self.strike)}

        tau = (self.expiry - today) / 250.

        call = blackScholes(tau, underlying, self.strike, vola)

        return {'delta': self.N * call['delta'],
                'npv': self.N * call['npv'],
                'vega': self.N * call['vega'],
                'gamma': self.N * call['gamma'],
                'theta': self.N * call['theta'],
                'intrinsic': self.N * max(0, underlying - self.strike)}
