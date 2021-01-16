import numpy as np
import pickle as pkl
from stable_baselines.common.buffers import ReplayBuffer
import matplotlib.pyplot as plt
from sklearn.manifold.t_sne import TSNE

# data preprocessing
file_dir = "/data1/hh/replay_buffer.pkl"
with open(file_dir, "rb") as rb_file:
    replay_buffer = pkl.load(rb_file)

print("loaded successfully")
assert isinstance(replay_buffer, ReplayBuffer)
obs = [replay_buffer.storage[i][0] for i in range(len(replay_buffer))]
obs = np.array(obs)
data_size = 1000000
# replay_buffer = [0 for i in range(data_size)]
# theta = np.random.rand(data_size) * 2 * np.pi
# phi = np.random.rand(data_size) * np.pi
#
# obs = [np.cos(phi) * np.sin(theta), np.cos(phi) * np.cos(theta), np.cos(phi)]
# obs = np.array(obs).swapaxes(0,1)
# obs = np.random.rand(data_size, 3) * 2 -1
# obs = obs / np.linalg.norm(obs, axis=1,keepdims=True)

# data visualizations
# model = TSNE()
# data_0 = np.random.choice(len(replay_buffer), data_size)
# print("computing low dimension embedding ...")
#
# low_dim_data = model.fit_transform(obs[data_0])
# print("low dimension embedding successfully")
# plt.scatter(low_dim_data[:, 0], low_dim_data[:, 1],c=obs[data_0,-1])
# plt.show()

# compute distance
# sample n data pair
data_1 = np.random.choice(len(replay_buffer), data_size)
data_2 = np.random.choice(len(replay_buffer), data_size)

dist = np.linalg.norm(obs[data_1] - obs[data_2], axis=1)
dist = dist[dist > 0]
print("dist computing successfully")
n, bins, patches = plt.hist(dist,bins=20)
plt.show()

print("mean dist", np.mean(dist), "min dist", np.min(dist), "max dist", np.max(dist))
