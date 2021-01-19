from gym.envs.registration import register
from toy_env.Jumper import JumperEnv
# MuJoCo
# ----------------------------------------

register(
    id='Jumper-v0',
    entry_point='toy_env:JumperEnv',
    max_episode_steps=1000,
)
