#!/usr/bin/env python3
import os
from stable_baselines import logger, A2C,PPO2
from stable_baselines.a2c.a2c_repr import A2CRepr
from stable_baselines.common.cmd_util import make_atari, Monitor, wrap_deepmind, set_global_seeds, DummyVecEnv, \
    SubprocVecEnv, atari_arg_parser
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.a2c.rlgan_warpper import AtariRescale42x42, AtariNoisyBackground
from attn_toy.policies.attn_policy import AttentionPolicy


def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None,
                   start_index=0, allow_early_resets=True,
                   start_method=None, use_subprocess=False, variation="constant_rectangle"):
    """
    Create a wrapped, monitored VecEnv for Atari.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
        `num_env` > 1, `DummyVecEnv` is usually faster. Default: False
    :return: (VecEnv) The atari environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                          allow_early_resets=allow_early_resets)
            # env = AtariNoisyBackground(env)
            env = AtariRescale42x42(env, variation)
            return wrap_deepmind(env, **wrapper_kwargs)

        return _thunk

    set_global_seeds(seed)

    # When using one environment, no need to start subprocesses
    if num_env == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i + start_index) for i in range(num_env)])

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)],
                         start_method=start_method)


def train(env_id, num_timesteps, seed, policy, lr_schedule, num_env, variation, load_path, encoder_coef,decoder_coef=0.1,repr_coef=1.,
          use_attention=True, save_interval=100000,learning_rate=2.5e-4,vf_coef=0.5,begin_repr=1.):
    """
    Train A2C model for atari environment, for testing purposes

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    :param policy: (A2CPolicy) The policy model to use (MLP, CNN, LSTM, ...)
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                 'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param num_env: (int) The number of environments
    """
    policy_fn = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy,
                 'attention': AttentionPolicy}[policy]

    train_env = VecFrameStack(make_atari_env(env_id, num_env, seed, variation="standard"), 4)
    test_env = VecFrameStack(make_atari_env(env_id, num_env, seed, variation=variation), 4)

    if load_path is None:
        model = A2CRepr(policy_fn, train_env, test_env,learning_rate=learning_rate,vf_coef=vf_coef,lr_schedule=lr_schedule, 
        seed=seed, repr_coef=repr_coef,atten_encoder_coef=encoder_coef,atten_decoder_coef=decoder_coef,
                        verbose=1, use_attention=use_attention)
    else:
        model = A2CRepr.load(load_path=load_path)
        model.set_env(test_env)
        print("load model successfully")
    epochs = num_timesteps // save_interval
    save_path = os.path.join(os.getenv('OPENAI_LOGDIR'), "save")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for epoch in range(epochs):
        model.learn(total_timesteps=save_interval, reset_num_timesteps=epoch == 0, print_attention_map=True,
                    repr_coef=[repr_coef] if epoch > epochs * begin_repr else [0.])
        print(model.num_timesteps)
        model.eval(int(save_interval/10), print_attention_map=True, filedir=None)
        if epoch%200==2:
            model.save(os.path.join(save_path, "model_{}.pkl".format((epoch + 1) * save_interval)))
    # model.learn(total_timesteps=int(num_timesteps * 1.1))
    train_env.close()
    test_env.close()

def train_a2c(env_id, num_timesteps, seed, policy, num_env, variation,
          save_interval=100000):
    """
    Train A2C model for atari environment, for testing purposes

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    :param policy: (A2CPolicy) The policy model to use (MLP, CNN, LSTM, ...)
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                 'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param num_env: (int) The number of environments
    """
    policy_fn = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy,
                 'attention': AttentionPolicy}[policy]

    train_env = VecFrameStack(make_atari_env(env_id, num_env, seed, variation="standard"), 4)
    #test_env = VecFrameStack(make_atari_env(env_id, num_env, seed, variation=variation), 4)


    model = A2C(policy_fn, train_env,seed=seed,verbose=1,tensorboard_log='/home/lzy/atten_baselines/experiments/a2c')
    epochs = num_timesteps // save_interval
    save_path = os.path.join(os.getenv('OPENAI_LOGDIR'), "save")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for epoch in range(epochs):
        model.learn(total_timesteps=save_interval, reset_num_timesteps=epoch == 0)
        print(model.num_timesteps)
        if epoch%50==2:
            model.save(os.path.join(save_path, "model_{}.pkl".format((epoch + 1) * save_interval)))
    # model.learn(total_timesteps=int(num_timesteps * 1.1))
    train_env.close()

def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp', 'attention'],
                        default='attention')
    parser.add_argument('--variation',
                        choices=['standard', 'moving-square', 'constant-rectangle', 'green-lines', 'diagonals'],
                        default='green-lines', help='Env variation')
    parser.add_argument('--repr_coef', help='reprenstation loss coefficient', type=float, default=0.)
    parser.add_argument('--begin_repr', help='reprenstation loss coefficient', type=float, default=0.)
    parser.add_argument('--use-attention',help='if or not to use attention', type=int, default=1)

    parser.add_argument('--lr_schedule', choices=['constant', 'linear'], default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--vf_coef', default=0.25,type=float,
                        help='value loss coef')
    parser.add_argument('--lr', default=7e-4,type=float,
                        help='Learning rate')
    parser.add_argument('--encoder_coef', default= 0.0,type=float,
                        help='encoder_coef')#1./2560
    parser.add_argument('--decoder_coef', default= 0.1,type=float,
                        help='decoder_coef')#1./2560
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path to load model')
    args = parser.parse_args()
    logger.configure()
    # train_a2c(args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy='cnn', \
    #     num_env=16, variation=args.variation)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy, lr_schedule=args.lr_schedule,
          num_env=16, variation=args.variation, repr_coef=args.repr_coef,learning_rate=args.lr,load_path=args.load_path,
          use_attention=(args.use_attention!=0),begin_repr=args.repr_coef,vf_coef=args.vf_coef,encoder_coef=args.encoder_coef
          ,decoder_coef=args.decoder_coef)


if __name__ == '__main__':
    main()
