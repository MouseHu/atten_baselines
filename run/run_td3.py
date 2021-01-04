import argparse
import time
import os

import gym
import tensorflow as tf
import numpy as np
import json
from mpi4py import MPI

from stable_baselines import logger, bench
from stable_baselines.common.misc_util import set_global_seeds, boolean_flag
# from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy
from stable_baselines import TD3
from stable_baselines.td3.td3_mem import TD3Mem
from stable_baselines.td3.td3_hard_target import TD3HT
from stable_baselines.td3.td3_mem_update import TD3MemUpdate
from stable_baselines.td3.td3_mem_many import TD3MemUpdateMany
from stable_baselines.td3.td3_mem_residue import TD3MemResidue
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines.common.wrappers import TimestepWrapper, DelayedRewardWrapper,NHWCWrapper
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from toy_env import *
import dmc2gym

def run(env_id, seed, noise_type, layer_norm, evaluation, agent, delay_step, nminibatches=4, n_steps=128, **kwargs):
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
    env = gym.make(env_id)
    env = TimestepWrapper(env)
    env = DelayedRewardWrapper(env, delay_step)
    # env = dmc2gym.make(
    #     domain_name=kwargs["domain_name"],
    #     task_name=kwargs["task_name"],
    #     seed=seed,
    #     visualize_reward=False,
    #     from_pixels=(kwargs["encoder_type"] == 'pixel'),
    #     height=kwargs["pre_transform_image_size"],
    #     width=kwargs["pre_transform_image_size"],
    #     frame_skip=kwargs["action_repeat"]
    # )
    # env=NHWCWrapper(env)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(0)))
    print(env.observation_space,env.action_space)
    if evaluation:
        eval_env = gym.make(env_id)
        eval_env = TimestepWrapper(eval_env)
        eval_env = DelayedRewardWrapper(eval_env, delay_step)
        # eval_env = dmc2gym.make(
        #     domain_name=kwargs["domain_name"],
        #     task_name=kwargs["task_name"],
        #     seed=seed,
        #     visualize_reward=False,
        #     from_pixels=(kwargs["encoder_type"] == 'pixel'),
        #     height=kwargs["pre_transform_image_size"],
        #     width=kwargs["pre_transform_image_size"],
        #     frame_skip=kwargs["action_repeat"]
        # )
        # eval_env = NHWCWrapper(eval_env)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        eval_env = bench.Monitor(eval_env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
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

    # Seed everything to make things reproducible.
    logger.info('seed={}, logdir={}'.format(seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    start_time = time.time()

    if layer_norm:
        if kwargs["encoder_type"] == 'pixel':
            policy = 'TD3LnCnnPolicy'
        else:
            policy = 'TD3LnMlpPolicy'
    else:
        policy = 'TD3MlpPolicy'

    num_timesteps = kwargs['num_timesteps']
    del kwargs['num_timesteps']

    models = {"TD3": TD3, "TD3Mem": TD3Mem, "TD3MemResidue": TD3MemResidue, "PPO2": PPO2, "TD3MemUpdate": TD3MemUpdate,
              "TD3MemMany": TD3MemUpdateMany, "TD3HT": TD3HT}

    model_func = models.get(agent, TD3)
    if model_func == PPO2:
        print("using ppo")
        policy = MlpPolicy
        model = PPO2(policy=policy, env=env, n_steps=n_steps, nminibatches=nminibatches,
                     lam=0.95, gamma=0.99, noptepochs=10, ent_coef=.01,
                     learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=2)
    else:
        model = model_func(policy=policy, env=env, eval_env=eval_env,
                           action_noise=action_noise, buffer_size=int(1e5), verbose=2, n_cpu_tf_sess=10)
    model.learn(total_timesteps=num_timesteps)
    if models == TD3Mem:
        model.memory.plot()
        model.memory.save(os.getenv("OPENAI_LOGDIR"))
    else:
        model.replay_buffer.save(os.getenv("OPENAI_LOGDIR"))
    env.close()
    if eval_env is not None:
        eval_env.close()

    logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    """
    parse the arguments for DDPG training

    :return: (dict) the arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Ant-v2')
    parser.add_argument('--agent', type=str, default='TD3')
    # boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    # boolean_flag(parser, 'render', default=False)
    # boolean_flag(parser, 'normalize-returns', default=False)
    # boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=int(time.time()))
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

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


def save_args(args):
    log_dir = os.getenv("OPENAI_LOGDIR")
    os.makedirs(log_dir, exist_ok=True)
    param_file = os.path.join(log_dir, "params.txt")
    with open(param_file, "w") as pf:
        pf.write(json.dumps(args))


if __name__ == '__main__':
    args = parse_args()
    save_args(args)
    logger.configure()
    # Run actual script.
    run(**args)
