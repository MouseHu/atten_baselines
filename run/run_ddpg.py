import time
import os

import gym
import tensorflow as tf
import numpy as np
import json
from run.run_util import create_action_noise, create_env, parse_args

from stable_baselines import logger, bench
from stable_baselines.common.misc_util import set_global_seeds, boolean_flag
# from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy
# from stable_baselines.sac.policies import LnMlpPolicy as SACLnMlpPolicy

from stable_baselines.ddpg import DDPG


def run(env_type, env_id, seed, noise_type, layer_norm, evaluation, agent, delay_step, gamma=0.99, nminibatches=4,
        n_steps=128, **kwargs):
    """
    run the training of DDPG

    :param env_id: (str) the environment ID
    :param seed: (int) the initial random seed
    :param noise_type: (str) the wanted noises ('adaptive-param', 'normal' or 'ou'), can use multiple noise type by
        seperating them with commas
    :param layer_norm: (bool) use layer normalization
    :param evaluation: (bool) enable evaluation of DDPG training
    :param kwargs: (dict) extra keywords for the training.train function
    """

    # Create envs.
    env = create_env(env_type, env_id, delay_step, seed, kwargs, str(0))
    print(env.observation_space, env.action_space)
    if evaluation:
        eval_env = create_env(env_type, env_id, delay_step, seed + 1, kwargs, "eval_env")
    else:
        eval_env = None

    # Parse noise_type

    action_noise = create_action_noise(env, noise_type)
    # Seed everything to make things reproducible.
    logger.info('seed={}, logdir={}'.format(seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed + 1)

    # Disable logging for rank != 0 to avoid noise.
    start_time = time.time()

    if layer_norm:
        if kwargs["encoder_type"] == 'pixel':
            policy = 'LnCnnPolicy'
        else:
            policy = 'LnMlpPolicy'
    else:
        if kwargs["encoder_type"] == 'pixel':
            policy = 'CnnPolicy'
        else:
            policy = 'MlpPolicy'

    num_timesteps = kwargs['num_timesteps']
    del kwargs['num_timesteps']

    model = DDPG(policy=policy, env=env, eval_env=eval_env, gamma=gamma, nb_eval_steps=5, batch_size=64,
                 nb_train_steps=100,nb_rollout_steps=100,
                action_noise=action_noise, buffer_size=int(1e6), verbose=2, n_cpu_tf_sess=10)
    print("model building finished")
    model.learn(total_timesteps=num_timesteps)
    env.close()
    if eval_env is not None:
        eval_env.close()

    logger.info('total runtime: {}s'.format(time.time() - start_time))


def save_args(args):
    log_dir = os.getenv("OPENAI_LOGDIR")
    os.makedirs(log_dir, exist_ok=True)
    param_file = os.path.join(log_dir, "params.txt")
    with open(param_file, "w") as pf:
        pf.write(json.dumps(args))


if __name__ == '__main__':
    args = parse_args()
    os.environ["OPENAI_LOGDIR"] = os.path.join(os.getenv("OPENAI_LOGDIR"), args["comment"])
    save_args(args)
    logger.configure()
    # Run actual script.
    run(**args)
