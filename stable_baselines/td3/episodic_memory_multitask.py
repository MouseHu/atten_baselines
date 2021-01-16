import numpy as np
from stable_baselines.td3.episodic_memory import EpisodicMemory
import hnswlib
import os
import pickle as pkl

class EpisodicMemoryMultiTask(EpisodicMemory):
    """
    Enable using K-KNN with representation and multiple gamma for multitask setting
    """

    def __init__(self, buffer_size, state_dim, action_shape, obs_space, q_func, repr_func, obs_ph, action_ph, sess,
                 gamma=0.99, alpha=0.6, knn=4):

        gamma = [gamma] if isinstance(gamma, (float, np.float)) else gamma

        super(EpisodicMemoryMultiTask, self).__init__(buffer_size, state_dim, action_shape, obs_space, q_func,
                                                      repr_func, obs_ph, action_ph, sess, gamma[0], alpha)

        self.key_buffer = np.zeros((buffer_size, state_dim))
        self.gamma_set = gamma
        self.knn = knn
        self.p = None
        self._q_values_set = -np.inf * np.ones((len(self.gamma_set), buffer_size + 1))

    def save(self, filedir):
        save_dict = {"query_buffer": self.query_buffer, "returns": self.returns,
                     "replay_buffer": self.replay_buffer, "reward_buffer": self.reward_buffer,
                     "truly_done_buffer": self.truly_done_buffer, "next_id": self.next_id, "prev_id": self.prev_id,
                     "gamma": self.gamma, "_q_values": self._q_values, "done_buffer": self.done_buffer,
                     "curr_capacity": self.curr_capacity, "capacity": self.capacity,"gamma_set":self.gamma_set,
                     "key_buffer":self.key_buffer,"knn":self.knn,"_q_value_set":self._q_values_set}

        with open(os.path.join(filedir, "episodic_memory.pkl"), "wb") as memory_file:
            pkl.dump(save_dict, memory_file)

    def init_p_index(self):
        if self.curr_capacity == 0:
            return
        self.p = p = hnswlib.Index(space='ip', dim=self.state_dim)  # possible options are l2, cosine or ip
        p.init_index(max_elements=self.capacity, ef_construction=200, M=16)
        p.set_ef(50)
        # p.set_num_threads(4)
        p.add_items(self.key_buffer[:self.curr_capacity], np.arange(self.curr_capacity))

    def update_key(self, idxes, reprs):
        self.key_buffer[idxes] = reprs

    def update_query(self, idxes, reprs):
        self.query_buffer[idxes] = reprs

    @property
    def q_values(self):
        return self._q_values_set[0, :]

    def compute_approximate_multi_q_with_repr(self, idxs):
        obses = self.replay_buffer[idxs]
        qs, repr = self.sess.run([self.q_func, self.repr_func], feed_dict={self.obs_ph: obses})
        assert len(qs.shape) == 3 and qs.shape[-1] == len(self.gamma_set), qs.shape
        self.update_key(idxs, repr[0])
        self.update_query(idxs, repr[0])
        return np.min(qs, axis=0)

    def fetch_knn_q(self, idxs):
        labels, _ = self.p.knn_query(self.query_buffer[idxs], k=self.knn)
        knn_qs = [np.mean(self.returns[[l for l in label if l != id]]) for label,id in zip(labels,idxs)]
        return knn_qs

    def compute_approximate_return(self, obses, actions=None):
        qs = np.array(self.sess.run(self.q_func, feed_dict={self.obs_ph: obses}))
        assert len(qs.shape) == 3 and qs.shape[-1] == len(self.gamma_set), qs.shape
        return np.min(qs[:, :, 0], axis=0)

    def compute_approximate_multi_q(self, obses):
        return np.min(np.array(self.sess.run(self.q_func, feed_dict={self.obs_ph: obses})), axis=0)

    def update_memory(self, q_base=0, use_knn=False,beta=0.5):

        trajs = self.retrieve_trajectories()
        approximate_qs_list = []

        for traj in trajs:
            if use_knn:
                approximate_qs = self.compute_approximate_multi_q_with_repr(traj)
            else:
                approximate_qs = self.compute_approximate_multi_q(self.replay_buffer[traj])
            approximate_qs_list.append(approximate_qs.reshape(-1, len(self.gamma_set)))

        for approximate_qs, traj in zip(approximate_qs_list, trajs):
            approximate_qs = np.concatenate([np.zeros((1, len(self.gamma_set))), approximate_qs], axis=0)
            assert len(approximate_qs) - 1 == len(traj)

            for j, gamma in enumerate(self.gamma_set):
                Rtn = -1e10
                Rtn_true = 0
                for i, s in enumerate(traj):
                    approximate_q = self.reward_buffer[s] + gamma * (1 - self.truly_done_buffer[s]) * (
                            approximate_qs[i, j] - q_base)
                    Rtn = self.reward_buffer[s] + gamma * (1 - self.truly_done_buffer[s]) * Rtn
                    Rtn = max(Rtn, approximate_q)
                    if self.done_buffer[s]:
                        Rtn_true = approximate_qs[i]
                    else:
                        Rtn_true = self.reward_buffer[s] + gamma * (1 - self.truly_done_buffer[s]) * Rtn_true
                    self._q_values_set[j, s] = Rtn
                    self.returns[s] = Rtn_true
        theta = 0.5
        if use_knn:
            self.init_p_index()
            for traj in trajs:
                knn_q = self.fetch_knn_q(traj)
                for i, s in enumerate(traj):
                    self.q_values[s] = min(theta*self.q_values[s]+(1-theta)*knn_q[i],self.q_values[s])

    def sample(self, batch_size, mix=False,priority=False):
        batch = super(EpisodicMemoryMultiTask, self).sample(batch_size, mix,priority)
        if batch is None:
            return None
        batch_idxs = batch['index0']
        batch['multi_q'] = self._q_values_set[:, batch_idxs.reshape(-1)].transpose()
        return batch
