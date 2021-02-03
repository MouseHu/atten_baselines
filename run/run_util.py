import gym
import tensorflow as tf
import numpy as np
import time
import json
from mpi4py import MPI
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines.common.wrappers import TimestepWrapper, DelayedRewardWrapper, NHWCWrapper
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from toy_env import *
import dmc2gym
from stable_baselines import logger, bench
from stable_baselines.common.misc_util import set_global_seeds, boolean_flag
import argparse

def create_env(env_type, env_id, delay_step, seed, kwargs, env_str=str(0)):
    if env_type in ["mujoco", "Mujoco", "MuJoCo", "raw", "mujoco_raw", "raw_mujoco"]:
        env = gym.make(env_id)
        env = TimestepWrapper(env)
        env = DelayedRewardWrapper(env, delay_step)
    else:
        env = dmc2gym.make(
            domain_name=kwargs["domain_name"],
            task_name=kwargs["task_name"],
            seed=seed,
            visualize_reward=False,
            from_pixels=(kwargs["encoder_type"] == 'pixel'),
            height=kwargs["pre_transform_image_size"],
            width=kwargs["pre_transform_image_size"],
            frame_skip=kwargs["action_repeat"]
        )
        env = NHWCWrapper(env)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), env_str))
    return env


def create_action_noise(env, noise_type):
    action_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mean=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(nb_actions),
                                                        sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
    return action_noise


def parse_args():
    """
    parse the arguments for DDPG training

    :return: (dict) the arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_type', type=str, default="mujoco")
    parser.add_argument('--env-id', type=str, default='Ant-v2')
    parser.add_argument('--agent', type=str, default='TD3')
    # boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=False)
    # boolean_flag(parser, 'render', default=False)
    # boolean_flag(parser, 'normalize-returns', default=False)
    # boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=int(time.time()))
    parser.add_argument('--comment', help='to show name', type=str, default="show_name_in_htop")
    # parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=100)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'enable-popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    # parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    # parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    # parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--noise-type', type=str, default='normal_0.1')
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--nb-eval-steps', type=int, default=5)

    parser.add_argument('--delay-step', type=int, default=0)

    boolean_flag(parser, 'evaluation', default=False)

    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--encoder_type', default='vector', type=str)

    # ddq
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta', default=-1, type=float)
    parser.add_argument('--num_q', default=4, type=int)
    parser.add_argument('--policy_delay', type=int, default=2)
    parser.add_argument('--gradient_steps', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--max_step', type=int, default=1000)

    boolean_flag(parser, 'iterative_q', default=False)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args