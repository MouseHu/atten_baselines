from stable_baselines.td3.episodic_memory import EpisodicMemory
import numpy as np
import time


class EpisodicMemoryDDQ(EpisodicMemory):
    def __init__(self, buffer_size, state_dim, action_shape, obs_space, q_func, repr_func, obs_ph, action_ph, sess,
                 gamma=0.99,
                 alpha=0.6):
        super(EpisodicMemoryDDQ, self).__init__(buffer_size, state_dim, action_shape, obs_space, q_func, repr_func,
                                                obs_ph, action_ph, sess,
                                                gamma, alpha)
        del self._q_values
        self._q_values = -np.inf * np.ones((buffer_size + 1, 2))
        self.max_step = 1

    def compute_approximate_return_double(self, obses, actions=None):
        return np.array(self.sess.run(self.q_func, feed_dict={self.obs_ph: obses}))

    def update_memory(self, q_base=0, use_knn=False, beta=-1):
        discount_beta = beta ** np.arange(self.max_step)
        trajs = self.retrieve_trajectories()
        for traj in trajs:
            # print(np.array(traj))
            approximate_qs = self.compute_approximate_return_double(self.replay_buffer[traj], self.action_buffer[traj])
            if len(approximate_qs) == 4:
                approximate_qs = approximate_qs.reshape(2, 2, -1)
                approximate_qs = np.min(approximate_qs, axis=1)  # clip double q

            else:
                assert len(approximate_qs) == 2
                approximate_qs = approximate_qs.reshape(2, -1)
            approximate_qs = np.concatenate([np.zeros((2, 1)), approximate_qs], axis=1)

            self.q_values[traj] = 0
            rtn_1 = np.zeros((len(traj), len(traj)))
            rtn_2 = np.zeros((len(traj), len(traj)))

            for i, s in enumerate(traj):
                rtn_1[i, 0], rtn_2[i, 0] = self.reward_buffer[s] + \
                                           self.gamma * (1 - self.truly_done_buffer[s]) * (
                                                   approximate_qs[:, i] - q_base)
            for i, s in enumerate(traj):
                rtn_1[i, 1:] = self.reward_buffer[s] + self.gamma * rtn_1[i - 1, :-1]
                rtn_2[i, 1:] = self.reward_buffer[s] + self.gamma * rtn_2[i - 1, :-1]
            if beta > 0:

                double_rtn = [
                    [np.dot(rtn_2[i, :min(i + 1, self.max_step)], discount_beta[:min(i + 1, self.max_step)]) / np.sum(
                        discount_beta[:min(i + 1, self.max_step)]),
                     np.dot(rtn_1[i, :min(i + 1, self.max_step)], discount_beta[:min(i + 1, self.max_step)]) / np.sum(
                         discount_beta[:min(i + 1, self.max_step)])]
                    for i in range(len(traj))]
            else:
                double_rtn = [
                    [rtn_2[i, np.argmax(rtn_1[i, :min(i + 1, self.max_step)])],
                     rtn_1[i, np.argmax(rtn_2[i, :min(i + 1, self.max_step)])]] for i
                    in
                    range(len(traj))]
            # self.q_values[traj] = np.maximum(np.array(double_rtn),np.minimum(rtn_1[:,0],rtn_2[:,0]))
            one_step_q = np.array([rtn_1[:, 0], rtn_2[:, 0]]).transpose()
            self.q_values[traj] = np.maximum(np.array(double_rtn),
                                             np.min(one_step_q,axis=1,keepdims=True))
                                             # one_step_q)
