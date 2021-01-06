import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os
import gc
import pickle as pkl
import copy
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
import random
from sklearn.manifold.t_sne import TSNE
import matplotlib.pyplot as plt
import math


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class EpisodicMemory(object):
    def __init__(self, buffer_size, state_dim, action_dim, action_shape, obs_space, qfs, obs_ph, action_ph, sess,
                 gamma=0.99,
                 alpha=0.6):
        buffer_size = int(buffer_size)
        self.ec_buffer = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = buffer_size
        self.curr_capacity = 0
        self.pointer = 0
        self.obs_space = obs_space

        self.latent_buffer = np.zeros((buffer_size, state_dim + action_dim))
        self.q_values = -np.inf * np.ones(buffer_size + 1)
        self.returns = -np.inf * np.ones(buffer_size + 1)
        self.replay_buffer = np.empty((buffer_size,) + obs_space.shape, np.float32)
        self.action_buffer = np.empty((buffer_size,) + action_shape, np.float32)
        self.reward_buffer = np.empty((buffer_size,), np.float32)
        self.steps = np.empty((buffer_size,), np.int)
        self.done_buffer = np.empty((buffer_size,), np.bool)
        self.next_id = -1 * np.ones(buffer_size)
        self.prev_id = [[] for _ in range(buffer_size)]
        self.ddpg_q_values = -np.inf * np.ones(buffer_size)
        self.lru = np.zeros(buffer_size)
        self.time = 0
        self.gamma = gamma
        self.hashes = dict()
        self.reward_mean = None
        self.kd_tree = None
        self.state_kd_tree = None
        self.build_tree = False
        self.build_tree_times = 0
        self.min_return = 0
        self.end_points = []
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        self.qfs = qfs
        self.obs_ph = obs_ph
        self.action_ph = action_ph
        self.sess = sess

    def squeeze(self, obses):
        return np.array([(obs - self.obs_space.low) / (self.obs_space.high - self.obs_space.low) for obs in obses])

    def unsqueeze(self, obses):
        return np.array([obs * (self.obs_space.high - self.obs_space.low) + self.obs_space.low for obs in obses])

    def save(self, filedir):
        pkl.dump(self, open(os.path.join(filedir, "episodic_memory.pkl"), "wb"))

    def add(self, obs, action, state, encoded_action, sampled_return, next_id=-1):
        if state is not None and encoded_action is not None:
            state, encoded_action = np.squeeze(state), np.squeeze(encoded_action)
            if len(encoded_action.shape) == 0:
                encoded_action = encoded_action[np.newaxis, ...]
        index = self.pointer
        self.pointer = (self.pointer + 1) % self.capacity
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
            # priority = self.w_q *(self.q_values[:self.capacity]) + self.lru
            # priority = self.q_values[:self.capacity]
            # priority = self.lru
            # index = int(np.argmin(priority))
            # print("Switching out...")

            if index in self.end_points:
                self.end_points.remove(index)
            # index = int(np.argmin(self.q_values))
            self.prev_id[index] = []
            self.next_id[index] = -1
            self.q_values[index] = -np.inf
            old_key = tuple(
                np.squeeze(np.concatenate([self.replay_buffer[index], self.action_buffer[index]])).astype('float32'))
            self.hashes.pop(old_key, None)

        else:
            # index = self.curr_capacity
            self.curr_capacity = min(self.capacity, self.curr_capacity + 1)
        self.replay_buffer[index] = obs
        self.action_buffer[index] = action
        if state is not None and encoded_action is not None:
            self.latent_buffer[index] = np.concatenate([state, encoded_action])
        self.q_values[index] = sampled_return
        self.returns[index] = sampled_return
        self.lru[index] = self.time
        new_key = tuple(np.squeeze(np.concatenate([obs, action])).astype('float32'))
        self.hashes[new_key] = index

        if next_id >= 0:
            self.next_id[index] = next_id
            if index not in self.prev_id[next_id]:
                self.prev_id[next_id].append(index)
        self.time += 0.01
        return index

    def update_priority(self):
        priorities = self.q_values[:self.curr_capacity] - np.min(self.q_values[:self.curr_capacity])
        for idx, priority in enumerate(priorities):
            priority = max(priority, 1e-6)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def peek(self, action, state, value_decay, modify, next_id=-1, allow_decrease=False, knn=1):
        state = np.squeeze(state)
        if len(state.shape) == 1:
            state = state[np.newaxis, ...]
        action = np.squeeze(action)
        if len(action.shape) == 0:
            action = action[np.newaxis, ...]
        if len(action.shape) == 1:
            action = action[np.newaxis, ...]
        h = np.concatenate([state, action], axis=1)
        if self.curr_capacity == 0 or self.build_tree == False:
            return None, None
        # print(h.shape)
        dist, ind = self.kd_tree.query(h, k=knn)
        dist, ind = dist[0], ind[0]
        if dist[0] < 1e-5:
            self.lru[ind] = self.time
            self.time += 0.01
            if modify:
                # if value_decay > self.q_values[ind]:
                if allow_decrease:
                    self.q_values[ind] = value_decay
                else:
                    self.q_values[ind] = max(value_decay, self.q_values[ind])
                if next_id >= 0:
                    self.next_id[ind] = next_id
                    if ind not in self.prev_id[next_id]:
                        self.prev_id[next_id].append(ind)

            return self.q_values[ind], ind
        else:
            queried_q_value = 0.0
            dist = dist / (1e-12 + np.max(dist)) + 1e-13
            coeff = -np.log(dist)
            coeff = coeff / np.sum(coeff)
            for i, index in enumerate(ind):
                queried_q_value += self.q_values[index] * coeff[i]

            return queried_q_value, None

    def sample_neg_keys(self, avoids, batch_size):
        # sample negative keys
        assert batch_size + len(
            avoids) <= self.capacity, "can't sample that much neg samples from episodic memory!"
        places = []
        while len(places) < batch_size:
            ind = np.random.randint(0, self.curr_capacity)
            if ind not in places:
                places.append(ind)
        return places

    def update_repr(self, idxes, reprs):
        self.latent_buffer[idxes] = reprs

    def update_sequence_corrected(self, sequence):
        # print(sequence)
        next_id = -1
        Rtd = 0
        # Rtd_soft = [0]
        # for obs, a, z, encoded_a, r, q_tp1, done in reversed(sequence):
        #     Rtd_soft.append(self.gamma * Rtd_soft[-1] + r)
        # Q_soft = np.mean(Rtd_soft) / (1. - (1. - self.gamma ** len(sequence)) / ((1. - self.gamma) * len(sequence)))

        for obs, a, z, encoded_a, r, q_tp1, done in reversed(sequence):
            # print(np.mean(z))
            if done:
                Rtd = q_tp1 if q_tp1 is not None else 0
            else:
                Rtd = self.gamma * Rtd + r
            # obs = self.squeeze([obs])[0]
            # qd, current_id = self.peek(encoded_a, z, Rtd, True)
            # if current_id is None:  # new action
            current_id = self.add(obs, a, z, encoded_a, Rtd, next_id)

            self.replay_buffer[current_id] = obs
            self.reward_buffer[current_id] = r
            self.done_buffer[current_id] = done
            next_id = int(current_id)
        self.update_priority()
        return

    def compute_approximate_return(self, obses, actions=None):
        return np.min(np.array(self.sess.run(self.qfs, feed_dict={self.obs_ph: obses})), axis=0)

    def compute_statistics(self, batch_size=128):
        estimated_qs = []
        for i in range(math.ceil(self.curr_capacity / batch_size)):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.curr_capacity)
            obses = self.replay_buffer[start:end]
            actions = None
            estimated_qs.append(self.compute_approximate_return(obses, actions).reshape(-1))
        estimated_qs = np.concatenate(estimated_qs)
        diff = estimated_qs - self.q_values[:self.curr_capacity]
        return np.min(diff), np.mean(diff), np.max(diff)

    def update_return(self, batch_size=128):

        for i in range(math.ceil(self.curr_capacity / batch_size)):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.curr_capacity)
            index = self.next_id[start:end]
            protected_index = np.max(0, index)
            obses = self.replay_buffer[protected_index]
            actions = self.action_buffer[protected_index]
            # qs = sess.run(qfs, feed_dict={obs_ph: obses, action_ph: actions})
            # q_2 = sess.run(qf_2, feed_dict={obs_ph: obses, action_ph: actions})
            # target_q = np.squeeze(np.min(qs,axis=1))

            target_q = self.reward_buffer[start:end] + self.gamma * (
                    1 - self.done_buffer[start:end]) * self.compute_approximate_return(obses, actions).reshape(-1)
            target_q = target_q.reshape(-1)

            assert target_q.shape == self.q_values[start:end].shape, "shape mismatch {} vs {}".format(target_q.shape,
                                                                                                      self.q_values[
                                                                                                      start:end].shape)
            self.q_values[start:end] = np.maximum(target_q, self.q_values[start:end])

    def retrieve_trajectories(self):
        trajs = []
        for e in self.end_points:
            traj = []
            prev = e
            while prev is not None:
                traj.append(prev)
                try:
                    prev = self.prev_id[prev][0]
                    # print(e,prev)
                except IndexError:
                    prev = None
            # print(np.array(traj))
            trajs.append(np.array(traj))
        return trajs

    def update_memory(self, q_base=0):
        np.set_printoptions(threshold=1200)
        trajs = self.retrieve_trajectories()

        for traj in trajs:
            # print(np.array(traj))
            approximate_qs = self.compute_approximate_return(self.replay_buffer[traj], self.action_buffer[traj])
            approximate_qs = approximate_qs.reshape(-1)
            assert approximate_qs.shape == traj.shape, "shape mismatch {} vs {}".format(approximate_qs.shape,
                                                                                        traj.shape)
            Rtn = 0
            approximate_qs = np.insert(approximate_qs, 0, 0)
            for i, s in enumerate(traj):
                approximate_q = self.reward_buffer[s] + self.gamma * (1 - self.done_buffer[s]) * (
                        approximate_qs[i] - q_base)
                Rtn = self.reward_buffer[s] + self.gamma * (1 - self.done_buffer[s]) * Rtn
                Rtn = max(Rtn, approximate_q)
                # self.q_values[s] = max(Rtn, approximate_q)
                self.q_values[s] = max(approximate_q,Rtn)
                # self.q_values[s] = approximate_q
                # self.q_values[s] = self.reward_buffer[s] + self.gamma * (1 - self.done_buffer[s]) * (
                #                 approximate_qs[i] - q_base)

    def update_sequence_with_qs(self, sequence, q_base=0):
        # print(sequence)
        next_id = -1
        Rtd = 0
        for obs, a, z, encoded_a, q_t, r, q_tp1, done in reversed(sequence):
            # print(np.mean(z))
            if done:
                Rtd = q_tp1 - q_base if q_tp1 is not None else r
            else:
                # Rtd = max(self.gamma * Rtd + r, q_t - q_base)
                Rtd = self.gamma * Rtd + r
            # obs = self.squeeze([obs])[0]
            # qd, current_id = self.peek(encoded_a, z, Rtd, True)
            # if current_id is None:  # new action
            current_id = self.add(obs, a, z, encoded_a, Rtd, next_id)
            if done:
                self.end_points.append(current_id)
            self.replay_buffer[current_id] = obs
            self.reward_buffer[current_id] = r
            self.done_buffer[current_id] = done
            next_id = int(current_id)
        # self.update_priority()
        return

    def update_kdtree(self, use_repr=True):
        if self.build_tree:
            del self.kd_tree
            del self.state_kd_tree
            # del self.hash_tree
        if self.curr_capacity <= 0:
            return
        # print("build tree", self.curr_capacity)
        # self.tree = KDTree(self.states[:self.curr_capacity])
        self.kd_tree = KDTree(self.latent_buffer[:self.curr_capacity])
        if use_repr:
            self.state_kd_tree = KDTree(self.latent_buffer[:self.curr_capacity, :self.state_dim])
        else:
            self.state_kd_tree = KDTree(self.replay_buffer[:self.curr_capacity])
        # self.hash_tree = KDTree(self.hashes[:self.curr_capacity])
        self.build_tree = True
        self.build_tree_times += 1
        if self.build_tree_times == 50:
            self.build_tree_times = 0
            gc.collect()

    def sample_negative(self, batch_size, batch_idxs, batch_idxs_next, batch_idx_pre):
        neg_batch_idxs = []
        i = 0
        while i < batch_size:
            neg_idx = np.random.randint(0, self.curr_capacity - 2)
            if neg_idx != batch_idxs[i] and neg_idx != batch_idxs_next[i] and neg_idx not in batch_idx_pre[i]:
                neg_batch_idxs.append(neg_idx)
                i += 1
        neg_batch_idxs = np.array(neg_batch_idxs)
        return neg_batch_idxs, self.replay_buffer[neg_batch_idxs]

    def switch_first_half(self, obs0, obs1, batch_size):
        tmp = copy.copy(obs0[:batch_size // 2, ...])
        obs0[:batch_size // 2, ...] = obs1[:batch_size // 2, ...]
        obs1[:batch_size // 2, ...] = tmp
        return obs0, obs1

    def sample(self, batch_size, mix=False):
        # Draw such that we always have a proceeding element
        if self.curr_capacity < batch_size:
            return None
        batch_idxs = []
        batch_idxs_next = []
        while len(batch_idxs) < batch_size:
            rnd_idx = np.random.randint(0, self.curr_capacity)
            # mass = random.random() * self._it_sum.sum(0, self.curr_capacity)
            # rnd_idx = self._it_sum.find_prefixsum_idx(mass)
            batch_idxs.append(rnd_idx)
            if self.next_id[rnd_idx] == -1:
                # be careful !!!!!! I use random id because in our implementation obs1 is never used
                if len(self.prev_id[rnd_idx]) > 0:
                    batch_idxs_next.append(self.prev_id[rnd_idx][0])
                else:
                    batch_idxs_next.append(0)
            else:
                batch_idxs_next.append(self.next_id[rnd_idx])

        batch_idxs = np.array(batch_idxs).astype(np.int)
        batch_idxs_next = np.array(batch_idxs_next).astype(np.int)
        batch_idx_pre = [self.prev_id[id] for id in batch_idxs]

        obs0_batch = self.replay_buffer[batch_idxs]
        obs1_batch = self.replay_buffer[batch_idxs_next]
        batch_idxs_neg, obs2_batch = self.sample_negative(batch_size, batch_idxs, batch_idxs_next, batch_idx_pre)
        action_batch = self.action_buffer[batch_idxs]
        reward_batch = self.reward_buffer[batch_idxs]
        terminal1_batch = self.done_buffer[batch_idxs]
        q_batch = self.q_values[batch_idxs]

        if mix:
            obs0_batch, obs1_batch = self.switch_first_half(obs0_batch, obs1_batch, batch_size)
        # obs0_batch, obs1_batch, obs2_batch = self.unsqueeze(obs0_batch), self.unsqueeze(obs1_batch), self.unsqueeze(
        #     obs2_batch)
        result = {
            'index0': array_min2d(batch_idxs),
            'index1': array_min2d(batch_idxs_next),
            'index2': array_min2d(batch_idxs_neg),
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'obs2': array_min2d(obs2_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
            'return': array_min2d(q_batch),
        }
        return result

    def plot(self):
        X = self.replay_buffer[:self.curr_capacity]
        model = TSNE()
        low_dim_data = model.fit_transform(X)
        plt.scatter(low_dim_data[:, 0], low_dim_data[:, 1])
        plt.show()
