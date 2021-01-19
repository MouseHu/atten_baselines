import time
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.math_util import safe_mean, unscale_action, scale_action
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.td3.policies import TD3Policy
from stable_baselines.td3.policies import CURL
import os
# from stable_baselines.td3.episodic_memory import EpisodicMemory
from stable_baselines.td3.episodic_memory_multitask import EpisodicMemoryMultiTask
from collections import deque


class TD3MemRepr(OffPolicyRLModel):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/pdf/1802.09477.pdf
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: (TD3Policy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values and Actor networks)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update" of the target networks, between 0 and 1)
    :param policy_delay: (int) Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param action_noise: (ActionNoise) the action noise type. Cf DDPG for the different action noise type.
    :param target_policy_noise: (float) Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: (float) Limit for absolute value of target policy smoothing noise.
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param gradient_steps: (int) How many gradient update after each step
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for TD3 normally but can help exploring when using HER + TD3.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on TD3 logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, eval_env, gamma=0.99, learning_rate=3e-4,
                 buffer_size=50000,
                 learning_starts=100, train_freq=100, gradient_steps=100, batch_size=128,
                 tau=0.005, policy_delay=2, contrastive_train_freq=1, action_noise=None,
                 nb_eval_steps=1000,alpha=0.5,beta=-1,num_q=4,iterative_q=True,
                 latent_dim=32,
                 target_policy_noise=0.2, target_noise_clip=0.5, start_policy_learning=0,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 separate_repr=True,
                 _init_setup_model=True, policy_kwargs=None, gamma_set=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        super(TD3MemRepr, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                         policy_base=TD3Policy, requires_vec_env=False,
                                         policy_kwargs=policy_kwargs,
                                         seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        print("TD3 Repr Agent here")
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.latent_dim = latent_dim
        self.gamma_set = [gamma] if gamma_set is None else gamma_set
        # self.gamma_set = [gamma, 0.999, 0.95, 0.9, 0.8] if gamma_set is None else gamma_set
        self.start_policy_learning = start_policy_learning
        self.action_noise = action_noise
        self.random_exploration = random_exploration
        self.policy_delay = policy_delay
        self.contrastive_train_freq = contrastive_train_freq
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.eval_env = eval_env
        self.nb_eval_steps = nb_eval_steps
        self.separate_repr = separate_repr
        self.alpha = alpha
        self.beta = beta
        self.num_q = num_q
        self.iterative_q=iterative_q

        self.graph = None
        self.replay_buffer = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy_tf = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.step_ops = None
        self.target_ops = None
        self.infos_names = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.policy_out = None
        self.policy_train_op = None
        self.policy_loss = None
        self.contrastive_loss = None
        self.contrastive_train_op = None

        self.memory = None
        # self.state_repr_func = state_repr_func
        # self.action_repr_func = action_repr_func
        self.qf1_pi = None
        self.qf2_pi = None
        self.qf3_pi = None
        self.qf4_pi = None
        self.qf1_target = None
        self.qf2_target = None
        self.qf3_target = None
        self.qf4_target = None
        self.qvalues_ph = None

        self.qf1_target_no_pi = None
        self.qf2_target_no_pi = None
        self.qf3_target_no_pi = None
        self.qf4_target_no_pi = None

        self.actions_pos_ph = None
        self.actions_neg_ph = None
        self.actions_tar_ph = None

        self.processed_pos_obs_ph = None
        self.processed_neg_obs_ph = None
        self.processed_tar_obs_ph = None

        self.pos_observations_ph = None
        self.neg_observations_ph = None
        self.tar_observations_ph = None
        self.qfs_target = None
        self.qfs_target_set = None

        self.next_observations_ph = None

        self.state_repr_t = None
        self.action_repr_t = None

        self.target_buffer_diff = None
        self.q_backup = None

        self.qfs = None
        self.qfs_pi = None
        self.qfs_target = None
        self.qfs_target_no_pi = None
        self.repr_save = None

        self.curl = None

        self.sequence = []
        self.q_base = 0
        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        policy_out = unscale_action(self.action_space, self.policy_out)
        return policy.obs_ph, self.actions_ph, policy_out

    @staticmethod
    def contrastive_loss_fc(emb_cur, emb_next, emb_neg, margin=1, c_type='margin'):
        def emb_dist(a, b):
            return tf.maximum(0., tf.reduce_sum(tf.squared_difference(a, b), axis=1))

        if c_type is None or c_type == 'origin':
            return tf.reduce_mean(
                tf.maximum(emb_dist(emb_cur, emb_next) - emb_dist(emb_cur, emb_neg) + margin, 0))
        elif c_type == 'sqmargin':
            return tf.reduce_mean(emb_dist(emb_cur, emb_next) +
                                  tf.maximum(0.,
                                             margin - emb_dist(emb_cur, emb_neg)))
        elif c_type == 'margin':
            return tf.reduce_mean(tf.math.sqrt(emb_dist(emb_cur, emb_next))
                                  + tf.square(tf.maximum(0., margin - tf.math.sqrt(emb_dist(emb_cur, emb_neg)))))

    def setup_model(self):
        # print("setup model ",self.observation_space.shape)
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                        **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    if self.separate_repr:
                        self.pos_observations_ph = self.policy_tf.pos_obs_ph
                        # self.neg_observations_ph = self.policy_tf.neg_obs_ph
                        self.tar_observations_ph = self.policy_tf.tar_obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    # self.processed_pos_obs_ph = self.policy_tf.processed_pos_obs
                    # self.processed_neg_obs_ph = self.policy_tf.processed_neg_obs
                    # self.processed_tar_obs_ph = self.policy_tf.processed_tar_obs
                    self.next_observations_ph = self.target_policy_tf.obs_ph
                    self.processed_next_obs_ph = self.target_policy_tf.processed_obs
                    self.action_target = self.target_policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.qvalues_ph = tf.placeholder(tf.float32, shape=(None, len(self.gamma_set)),
                                                     name='qvalues')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.weight_ph = tf.placeholder(tf.float32, shape=(self.batch_size,), name="contrast_weight_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    self.policy_out = policy_out = self.policy_tf.make_actor(self.processed_obs_ph)
                    critic_func = self.policy_tf.make_set_many_critics_repr

                    self.qfs, qfs_set, self.repr_save = critic_func(self.processed_obs_ph, self.actions_ph,
                                                                    latent_dim=self.latent_dim,
                                                                    num_gamma=len(self.gamma_set),
                                                                    scope="buffer_values_fn")

                    self.qfs_pi, _, repr_tar = critic_func(self.processed_obs_ph, policy_out,
                                                           scope="buffer_values_fn",
                                                           latent_dim=self.latent_dim,
                                                           num_gamma=len(self.gamma_set),
                                                           reuse=True)

                    if self.separate_repr:
                        _, _, self.repr_save = critic_func(self.processed_obs_ph, self.actions_ph,
                                                           latent_dim=self.latent_dim,
                                                           num_gamma=len(self.gamma_set),
                                                           scope="buffer_values_fn_separate", reuse=False)
                        _, _, repr_tar = critic_func(self.processed_obs_ph, policy_out,
                                                     scope="buffer_values_fn_separate",
                                                     latent_dim=self.latent_dim,
                                                     num_gamma=len(self.gamma_set),
                                                     reuse=True)
                    self.curl = CURL(self.sess, self.latent_dim, self.batch_size)

                with tf.variable_scope("target", reuse=False):
                    # Create target networks
                    critic_func = self.target_policy_tf.make_set_many_critics_repr

                    target_policy_out = self.target_policy_tf.make_actor(self.processed_next_obs_ph)

                    qfs_target, qfs_target_set, repr_pos = critic_func(self.processed_next_obs_ph,
                                                                       target_policy_out,
                                                                       scope="buffer_values_fn",
                                                                       num_gamma=len(self.gamma_set), reuse=False)

                    self.qfs_target = qfs_target
                    self.qfs_target_set = qfs_target_set
                    self.qfs_target_no_pi, _, _ = critic_func(self.processed_obs_ph, self.actions_ph,
                                                              latent_dim=self.latent_dim, num_gamma=len(self.gamma_set),
                                                              scope="buffer_values_fn", reuse=True)

                    if self.separate_repr:
                        _, _, repr_pos = critic_func(self.processed_next_obs_ph,
                                                     target_policy_out,
                                                     scope="buffer_values_fn_separate",
                                                     num_gamma=len(self.gamma_set), reuse=False)

                    # repr_pos_key = self.curl.compute_key(repr_pos)
                    repr_pos_key = repr_pos
                with tf.variable_scope("loss", reuse=False):
                    # Contrastive loss: learning representation

                    contrastive_loss = 0
                    for emb_tar, emb_pos in zip(repr_tar, repr_pos):
                        contrastive_loss += self.curl.compute_loss(emb_tar, tf.stop_gradient(emb_pos), self.weight_ph)

                    self.contrastive_loss = contrastive_loss
                    # Q value loss
                    # Take the min of the two target Q-Values (clipped Double-Q Learning)
                    min_qf_target = tf.reduce_mean(qfs_target, axis=0) - self.q_base

                    # Targets for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * min_qf_target
                    )
                    self.q_backup = q_backup
                    # Compute Q-Function loss
                    alpha = .5
                    qfs_loss = tf.reduce_mean(
                        tf.nn.leaky_relu(-self.qvalues_ph + qfs_set - self.q_base, alpha=alpha) ** 2)

                    target_buffer_diff = tf.reduce_mean((self.qvalues_ph[:, 0] - q_backup) ** 2)
                    self.target_buffer_diff = target_buffer_diff

                    qvalues_losses = qfs_loss
                    self.policy_loss = policy_loss = -tf.reduce_mean(self.qfs_pi)

                    # Policy loss: maximise q value

                    # Policy train op
                    # will be called only every n training steps,
                    # where n is the policy delay
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss,
                                                                var_list=tf_util.get_trainable_vars('model/pi'))
                    self.policy_train_op = policy_train_op

                    # Q Values optimizer
                    qvalues_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    qvalues_params = tf_util.get_trainable_vars('model/buffer_values_fn/')

                    contrastive_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    contrast_param_name = 'model/buffer_values_fn{}'.format('_separate' if self.separate_repr else '')
                    contrastive_params = tf_util.get_trainable_vars(contrast_param_name)
                    self.contrastive_train_op = contrastive_optimizer.minimize(contrastive_loss,
                                                                               var_list=contrastive_params)
                    # Q Values and policy target params
                    source_params = tf_util.get_trainable_vars("model/")
                    target_params = tf_util.get_trainable_vars("target/")

                    # Polyak averaging for target variables
                    self.target_ops = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # self.target_ops = [
                    #     tf.assign(target,
                    #               (1 - self.tau) ** self.gradient_steps * target +
                    #               (1 - (1 - self.tau) ** self.gradient_steps) * source)
                    #     for target, source in zip(target_params, source_params)
                    # ]

                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    train_values_op = qvalues_optimizer.minimize(qvalues_losses, var_list=qvalues_params)

                    self.infos_names = ['qfs_loss', 'target_buffer_diff', 'contrastive_loss']
                    # All ops to call during one training step
                    self.step_ops = [qfs_loss, target_buffer_diff,
                                     self.qfs, train_values_op, self.target_ops]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qfs_loss', qfs_loss)
                    # tf.summary.scalar('contrastive_loss', contrastive_loss)
                    tf.summary.scalar('target_buffer_diff', target_buffer_diff)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                print(self.params)
                self.target_params = tf_util.get_trainable_vars("target/")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

                self.memory = EpisodicMemoryMultiTask(int(1e5), state_dim=self.latent_dim, gamma=self.gamma_set,
                                                      obs_space=self.observation_space,
                                                      action_shape=self.action_space.shape,
                                                      q_func=self.qfs_target_set, repr_func=repr_pos_key,
                                                      obs_ph=self.processed_next_obs_ph,
                                                      action_ph=self.actions_ph, sess=self.sess)

    def _train_step(self, step, writer, learning_rate, update_policy, update_contrastive, update_q=True):
        # Sample a batch from the replay buffer
        # batch = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        batch = self.memory.sample(self.batch_size, mix=False)

        if batch is None:
            return 0, 0
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_returns = batch['obs0'], batch[
            'actions'], batch['rewards'], batch['obs1'], batch['terminals1'], batch['multi_q']

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate,
            self.qvalues_ph: batch_returns.reshape(self.batch_size, -1),
        }
        # print("training ",batch_obs.shape)
        step_ops = self.step_ops
        if update_policy:
            # Update policy and target networks
            step_ops = step_ops + [self.policy_train_op, self.policy_loss]
        if update_q:
            # Do one gradient step
            # and optionally compute log for tensorboard
            if writer is not None:
                out = self.sess.run([self.summary] + step_ops, feed_dict)
                summary = out.pop(0)
                writer.add_summary(summary, step)
            else:
                out = self.sess.run(step_ops, feed_dict)
            # Unpack to monitor losses
            qfs_loss, target_buffer_diff, *_values = out
        else:
            qfs_loss, target_buffer_diff = 0, 0
        if update_contrastive:
            batch_contra = self.memory.sample(self.batch_size, mix=True,priority=False)
            batch_tar_obs, batch_pos_obs,batch_count = batch_contra['obs0'], batch_contra['obs1'],batch_contra['count']
            contra_feed_dict = {
                self.weight_ph: np.ones(len(batch_count))/len(batch_count),
                # self.weight_ph: batch_count.reshape(-1)/np.sum(batch_count),
                self.learning_rate_ph: learning_rate,
                self.next_observations_ph: batch_pos_obs,
                self.observations_ph: batch_tar_obs,
            }
            contra_ops = [self.contrastive_train_op, self.contrastive_loss]
            _, contrastive_loss = self.sess.run(contra_ops, contra_feed_dict)
        else:
            contrastive_loss = 0

        return qfs_loss, target_buffer_diff, contrastive_loss

    def learn(self, total_timesteps, eval_interval=10000, update_interval=10000, callback=None,
              log_interval=4, tb_log_name="TD3", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)
            qs_buffer = deque(maxlen=1000)
            # q4s = deque(maxlen=1000)
            start_time = time.time()
            episode_rewards = [0.0]
            current_steps = 0
            episode_successes = []
            discount_episodic_reward = 0.
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()
            n_updates = 0
            infos_values = []

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            for step in range(total_timesteps):
                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if self.num_timesteps < self.learning_starts or np.random.rand() < self.random_exploration:
                    # actions sampled from action space are from range specific to the environment
                    # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                    unscaled_action = self.env.action_space.sample()
                    action = scale_action(self.action_space, unscaled_action)
                else:
                    action = self.policy_tf.step(obs[None]).flatten()
                    # Add noise to the action, as the policy
                    # is deterministic, this is required for exploration
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # Rescale from [-1, 1] to the correct bounds
                    unscaled_action = unscale_action(self.action_space, action)
                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(unscaled_action)
                truly_done = info.get('truly_done', done)
                self.num_timesteps += 1

                # Only stop training if return value is False, not when it is None. This is for backwards
                # compatibility with callbacks that have no return statement.
                callback.update_locals(locals())
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, reward

                qs, state_repr = self.sess.run([self.qfs, self.repr_save],
                                               feed_dict={self.processed_obs_ph: obs[None], self.actions_ph: [action]})
                q = np.squeeze(np.min(qs))
                qs_buffer.extend([q])
                # discount_episodic_reward = reward_ + self.gamma * discount_episodic_reward
                discount_episodic_reward += reward_ * (self.gamma ** current_steps)
                current_steps += 1
                # Store transition in the replay buffer.
                # print(state_repr)
                self.replay_buffer_add(obs_, action, reward_, new_obs_, done, info)
                self.sequence.append(
                    (obs_, action, state_repr[0], q, reward_, truly_done, False))

                if done:
                    action = self.policy_tf.step(new_obs[None]).flatten()
                    state_repr = self.sess.run(self.repr_save,
                                               feed_dict={self.processed_obs_ph: new_obs[None],
                                                          self.actions_ph: [action]})
                    self.sequence.append(
                        (new_obs_, action, state_repr[0], 0, 0, True, True))
                    self.memory.update_sequence_with_qs(self.sequence)
                    self.sequence = []

                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    self.ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_reward,
                                                        ep_done, writer, self.num_timesteps)

                if self.num_timesteps % self.train_freq == 0:
                    callback.on_rollout_end()
                    # self.sess.run(self.target_ops)
                    self.memory.update_memory(self.q_base, use_knn=True)
                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                                or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)

                        # if_train_policy = ((step + grad_step) % self.policy_delay == 0)
                        if_train_policy = (step > self.start_policy_learning) and \
                                          ((step + grad_step) % self.policy_delay == 0)
                        # if_train_contrastive = ((step + grad_step) % self.contrastive_train_freq == 0)
                        if_train_contrastive = True
                        if_train_q = True
                        mb_infos_vals.append(
                            self._train_step(step, writer, current_lr, if_train_policy, if_train_contrastive,
                                             if_train_q))

                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                    callback.on_rollout_start()
                #
                # if step == 300000:
                #     self.memory.clean()
                if step % eval_interval == 0:
                    # Evaluate.
                    eval_episode_rewards = []
                    eval_qs = []
                    if self.eval_env is not None:
                        eval_episode_reward = 0.
                        for _ in range(self.nb_eval_steps):
                            if step >= total_timesteps:
                                return self

                            eval_action = self.policy_tf.step(obs[None]).flatten()
                            unscaled_action = unscale_action(self.action_space, eval_action)
                            eval_obs, eval_r, eval_done, eval_info = self.eval_env.step(unscaled_action)
                            eval_episode_reward += eval_r

                            # Retrieve reward and episode length if using Monitor wrapper
                            eval_maybe_ep_info = eval_info.get('episode')
                            if eval_maybe_ep_info is not None:
                                self.eval_ep_info_buf.extend([eval_maybe_ep_info])

                            if eval_done:
                                if not isinstance(self.env, VecEnv):
                                    eval_obs = self.eval_env.reset()
                                eval_episode_rewards.append(eval_episode_reward)
                                eval_episode_reward = 0.
                        if len(eval_episode_rewards[-101:-1]) == 0:
                            eval_mean_reward = -np.inf
                        else:
                            eval_mean_reward = round(float(np.mean(eval_episode_rewards[-101:-1])), 1)

                        logger.logkv("eval mean 100 episode reward", eval_mean_reward)
                        if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                            logger.logkv('eval_ep_rewmean',
                                         safe_mean([ep_info['r'] for ep_info in self.eval_ep_info_buf]))
                            logger.logkv('eval_eplenmean',
                                         safe_mean([ep_info['l'] for ep_info in self.eval_ep_info_buf]))
                        logger.logkv('eval_time_elapsed', int(time.time() - start_time))
                        logger.dumpkvs()

                episode_rewards[-1] += reward_
                if done:
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                # substract 1 as we appended a new term just now
                num_episodes = len(episode_rewards) - 1
                if step % 200000 == 0:  # save every 200k step
                    self.save(os.path.join(os.getenv("OPENAI_LOGDIR"), "model.pkl"))
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and num_episodes % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv('qs_mean', safe_mean([x for x in qs_buffer]))
                    # logger.logkv('q4_mean', safe_mean([x for x in q4s]))
                    logger.logkv('discount_q', discount_episodic_reward)
                    logger.logkv('qs_difference',
                                 safe_mean([x for x in qs_buffer]) - discount_episodic_reward - self.q_base)
                    logger.logkv('qs_abs_difference',
                                 safe_mean([abs(x - discount_episodic_reward - self.q_base) for x in qs_buffer]))
                    # logger.logkv('q4_difference', safe_mean([x for x in q4s]) - discount_episodic_reward)
                    diff_min, diff_mean, diff_max = self.memory.compute_statistics()
                    logger.logkv("buffer_diff_min", diff_min)
                    logger.logkv("buffer_diff_mean", diff_mean)
                    logger.logkv("buffer_diff_max", diff_max)
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    qs_buffer.clear()
                    # q4s.clear()
                    discount_episodic_reward = 0.
                    current_steps = 0
                    infos_values = []

            callback.on_training_end()
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        _ = np.array(observation)

        if actions is not None:
            raise ValueError("Error: TD3 does not have action probabilities.")

        # here there are no action probabilities, as DDPG does not use a probability distribution
        warnings.warn("Warning: action probability is meaningless for TD3. Returning None")
        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation)

        if self.action_noise is not None and not deterministic:
            actions = np.clip(actions + self.action_noise(), -1, 1)

        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.action_space, actions)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "policy_delay": self.policy_delay,
            "target_noise_clip": self.target_noise_clip,
            "target_policy_noise": self.target_policy_noise,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
