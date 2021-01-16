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
from stable_baselines.td3.episodic_memory import EpisodicMemory
from stable_baselines.td3.td3_mem import TD3Mem


class TD3MemResidue(TD3Mem):
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
                 tau=0.005, policy_delay=2, action_noise=None,
                 nb_eval_steps=1000,
                 target_policy_noise=0.2, target_noise_clip=0.5, start_policy_learning=10000,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        super(TD3MemResidue, self).__init__(policy, env, eval_env, gamma, learning_rate,
                                            buffer_size, learning_starts, train_freq, gradient_steps, batch_size,
                                            tau, policy_delay, action_noise, nb_eval_steps, target_policy_noise,
                                            target_noise_clip,
                                            start_policy_learning,
                                            random_exploration, verbose, tensorboard_log, _init_setup_model,
                                            policy_kwargs,
                                            full_tensorboard_log, seed, n_cpu_tf_sess)
        print("TD3 Memory Residue Agent here")
        self.residue_q1 = None
        self.residue_q2 = None
        self.residue_qf1_target = None
        self.residue_qf2_target = None

    def setup_model(self):
        # print("setup model ",self.observation_space.shape)
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.memory = EpisodicMemory(int(1e6), state_dim=1, action_dim=1,
                                             obs_space=self.observation_space,
                                             action_shape=self.action_space.shape)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                        **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy_tf.obs_ph
                    self.processed_next_obs_ph = self.target_policy_tf.processed_obs
                    self.action_target = self.target_policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.qvalues_ph = tf.placeholder(tf.float32, shape=(None, 1),
                                                     name='qvalues')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    self.policy_out = policy_out = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Use two Q-functions to improve performance by reducing overestimation bias
                    qf1, qf2 = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph)
                    residue_qf1, residue_qf2 = self.policy_tf.make_duel_critics(self.processed_obs_ph,
                                                                                self.actions_ph,
                                                                                scope="residue_values_fn")
                    residue_qf1_pi, residue_qf2_pi = self.policy_tf.make_duel_critics(self.processed_obs_ph,
                                                                                      self.actions_ph,
                                                                                      scope="residue_values_fn",
                                                                                      reuse=True)
                    # Q value when following the current policy
                    qf1_pi, qf2_pi = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                 policy_out, reuse=True)
                    # self.qf1 = qf1
                    # self.qf2 = qf2
                    self.qf1_pi = qf1_pi
                    self.qf2_pi = qf2_pi

                with tf.variable_scope("target", reuse=False):
                    # Create target networks
                    target_policy_out = self.target_policy_tf.make_actor(self.processed_next_obs_ph)
                    # Target policy smoothing, by adding clipped noise to target actions
                    target_noise = tf.random_normal(tf.shape(target_policy_out), stddev=self.target_policy_noise)
                    target_noise = tf.clip_by_value(target_noise, -self.target_noise_clip, self.target_noise_clip)
                    # Clip the noisy action to remain in the bounds [-1, 1] (output of a tanh)
                    noisy_target_action = tf.clip_by_value(target_policy_out + target_noise, -1, 1)
                    # Q values when following the target policy
                    qf1_target, qf2_target = self.target_policy_tf.make_critics(self.processed_next_obs_ph,
                                                                                noisy_target_action)
                    residue_qf1_target, residue_qf2_target = self.target_policy_tf.make_duel_critics(
                        self.processed_next_obs_ph, noisy_target_action, scope="residue_values_fn")
                    self.qf1_target = qf1_target
                    self.residue_qf1_target = residue_qf1_target
                    self.qf2_target = qf2_target
                    self.residue_qf2_target = residue_qf2_target

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two target Q-Values (clipped Double-Q Learning)
                    q1_target = residue_qf1_target + qf1_target
                    q2_target = residue_qf2_target + qf2_target
                    min_residue_qf_target = tf.minimum(q1_target, q2_target)

                    # Targets for Q value regression
                    residue_q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * min_residue_qf_target
                    )

                    # Compute Q-Function loss
                    qf1_loss = tf.reduce_mean((self.qvalues_ph - qf1) ** 2)
                    qf2_loss = tf.reduce_mean((self.qvalues_ph - qf2) ** 2)
                    residue_qf1_loss = tf.reduce_mean((residue_q_backup - residue_qf1) ** 2)
                    residue_qf2_loss = tf.reduce_mean((residue_q_backup - residue_qf2) ** 2)
                    qvalues_losses = qf1_loss + qf2_loss + residue_qf1_loss + residue_qf2_loss

                    # Policy loss: maximise q value
                    self.policy_loss = policy_loss = -tf.reduce_mean(qf1_pi + residue_qf1_pi)

                    # Policy train op
                    # will be called only every n training steps,
                    # where n is the policy delay
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss,
                                                                var_list=tf_util.get_trainable_vars('model/pi'))
                    self.policy_train_op = policy_train_op

                    # Q Values optimizer
                    qvalues_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    qvalues_params = tf_util.get_trainable_vars('model/values_fn/') + tf_util.get_trainable_vars(
                        'model/residue_values_fn/')
                    print(qvalues_params)
                    # Q Values and policy target params
                    source_params = tf_util.get_trainable_vars("model/")
                    target_params = tf_util.get_trainable_vars("target/")

                    # Polyak averaging for target variables
                    self.target_ops = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    train_values_op = qvalues_optimizer.minimize(qvalues_losses, var_list=qvalues_params)

                    self.infos_names = ['qf1_loss', 'qf2_loss', 'residue_qf1_loss', 'residue_qf2_loss']
                    # All ops to call during one training step
                    self.step_ops = [qf1_loss, qf2_loss, residue_qf1_loss, residue_qf2_loss,
                                     qf1, qf2, train_values_op]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('residue_qf1_loss', residue_qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('residue_qf2_loss', residue_qf2_loss)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                self.target_params = tf_util.get_trainable_vars("target/")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate, update_policy):
        # Sample a batch from the replay buffer
        # batch = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        batch = self.memory.sample(self.batch_size, mix=False)
        if batch is None:
            return 0, 0
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_returns = batch['obs0'], batch[
            'actions'], batch['rewards'], batch['obs1'], batch['terminals1'], batch['return']
        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate,
            self.qvalues_ph: batch_returns.reshape(self.batch_size, -1)
        }
        # print("training ",batch_obs.shape)
        step_ops = self.step_ops
        if update_policy:
            # Update policy and target networks
            step_ops = step_ops + [self.policy_train_op, self.target_ops, self.policy_loss]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(step_ops, feed_dict)

        # Unpack to monitor losses
        qf1_loss, qf2_loss, residue_qf1_loss, residue_qf2_loss, *_values = out

        return qf1_loss, qf2_loss, residue_qf1_loss, residue_qf2_loss
