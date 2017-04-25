from gym.envs.registration import register

register(
    id='bs-v0',
    entry_point='gym_bs.envs:EuropeanOptionEnv',
)
