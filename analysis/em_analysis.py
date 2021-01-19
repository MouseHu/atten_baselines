import matplotlib as mpl
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.use("TkAgg")
file_dir = "/data1/hh/ddq130_2/ant_amc_ddq_true_alpha_0.5_1/episodic_memory.pkl"
# file_dir = "/data1/hh/ddq/ant_amc_ddq_alpha_1_nodp_2/episodic_memory.pkl"
file_dir = "C:/Users/Mouse Hu/Desktop/CEC/data/ant_amc_ddq_alpha_0.5_iterative_0/episodic_memory.pkl"
# file_dir = "C:/Users/Mouse Hu/Desktop/CEC/data/episodic_memory.pkl"


with open(file_dir, "rb") as memory_file:
    memory = pkl.load(memory_file)
    print(memory.keys())

returns = memory["returns"]
q_values = memory["_q_values"]
states = memory["replay_buffer"]
done_buffer = memory["done_buffer"]
pos = states[:, 1]
# vel = states[:,5:7]
end_points = np.where(done_buffer == True)[0].tolist()

trajs = [np.arange(x, y) for x, y in zip(end_points[:-1], end_points[1:])]
for i in range(10):
    n = np.random.randint(0, len(trajs))
    fig = plt.figure()
    traj = list(reversed(trajs[n]))
    # ax = Axes3D(fig)
    # print(len(traj),q_values[traj,:0].shape)
    plt.scatter(np.arange(len(traj)), q_values[traj,0] / 500, )
    plt.scatter(np.arange(len(traj)), returns[traj] / 500)
    plt.scatter(np.arange(len(traj)), pos[traj])
    # plt.scatter(np.arange(len(traj)),np.sqrt(vel[traj,0]**2))
    plt.legend(["q_values", "return", "pos_z"])
    plt.show()
