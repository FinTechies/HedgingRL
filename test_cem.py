import gym
import gym.envs
import gym_bs
import numpy as np
import json, sys, os
from os import path
import pickle

gym.envs.register(id='bs-v3',
                  entry_point='gym_bs.envs:EuropeanOptionEnv',
                  kwargs={'t': 1000})

env = gym.make('bs-v3')
env = gym.wrappers.Monitor(env, "/tmp/gym-results/bs-v3", video_callable=False, write_upon_reset=True, force=True)

class DeterministicContinuousActionLinearPolicy(object):
    """
    Taken from https://gym.openai.com/evaluations/eval_sXJlX4GVQouaTYTkWemOA
    """

    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        dim_ac: dimension of action vector
        theta: flat vector of parameters
        """
        self.ac_space = ac_space
        dim_ob = ob_space.shape[0]
        dim_ac = ac_space.shape[0]
        # print(theta, len(theta), dim_ob, dim_ac)
        try:
            assert len(theta) == (dim_ob + 1) * dim_ac
        except:
            pass

        self.W = theta[0 : dim_ob * dim_ac].reshape(dim_ob, dim_ac)
        self.b = theta[dim_ob * dim_ac : None]

    def act(self, ob):
        a = np.clip(ob.dot(self.W) + self.b, self.ac_space.low, self.ac_space.high)
        return a


def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function

    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in th_std[None,:] * np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.flatten().argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0).flatten()
        th_std = elite_ths.std(axis=0).flatten()
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}

def do_rollout(agent, env, num_steps, render=False):
    total_reward, t = 0, 0
    ob = env.reset()
    done = False
    while not done:
    # for t in range(env.T):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_reward += reward
        t += 1
#         if render and t%3==0: env.render()
        if done: break
    return total_reward, t+1

# env = gym.make('bs-v3')
# env.seed(0)
# np.random.seed(0)
params = dict(n_iter=1000, batch_size=25, elite_frac = 0.2)
num_steps = 200

# You provide the directory to write to (can be an existing
# directory, but can't contain previous monitor results. You can
# also dump to a tempdir if you'd like: tempfile.mkdtemp().
# outdir = '/tmp/cem-agent-results'
# env = gym.wrappers.Monitor(env, outdir, force=True)

# Prepare snapshotting
# ----------------------------------------
def writefile(fname, s):
    with open(path.join('/tmp/cem-agent-results/', fname), 'w+') as fh:
        fh.write(s)
    # pass

info = {}
info['params'] = params
info['env_id'] = env.spec.id
# ------------------------------------------

def noisy_evaluation(theta):
    # print(theta)
    agent = DeterministicContinuousActionLinearPolicy(theta, env.observation_space, env.action_space)
    reward, T = do_rollout(agent, env, num_steps)
    return reward

rewards = []
# Train the agent, and snapshot each stage
for (i, iterdata) in enumerate(cem(noisy_evaluation, np.zeros(env.observation_space.shape[0]+1), **params)):
    print('Iteration %2i. Episode mean reward: %7.3f'%(i, iterdata['y_mean']))
    # print(iterdata['theta_mean'])
    print(iterdata[''])
    agent = DeterministicContinuousActionLinearPolicy(iterdata['theta_mean'], env.observation_space, env.action_space)
    do_rollout(agent, env, 200, render=False)
    writefile('agent-%.4i.pkl' % i, str(pickle.dumps(agent, -1)))

# Write out the env at the end so we store the parameters of this
# environment.
print(iterdata)
writefile('info.json', json.dumps(info))

env.close()