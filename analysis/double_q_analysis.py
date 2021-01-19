import numpy as np
import time

np.random.seed(int(time.time()))
ep_len = 1000000
num_action = 100
snr = 0.5
alpha = 0.33


def random_func(length):
    # return np.random.randn(length)
    # return np.random.rand(num_action)*2-1
    return np.random.beta(1, 1, num_action) - 1. / 2


x = np.random.rand(num_action)
print(np.max(x))
estimate_1 = []
truth_1 = []

estimate_2 = []
truth_2 = []

estimate_3 = []
for i in range(ep_len):
    Q = x + snr * random_func(num_action)
    Q1 = x + snr * random_func(num_action)
    Q2 = x + snr * random_func(num_action)
    estimate_1.append(np.max(Q))
    truth_1.append(x[np.argmax(Q)])

    estimate_2.append(Q1[np.argmax(Q2)])
    truth_2.append(x[np.argmax(Q1)])

    estimate_3.append((1-alpha)*Q1[np.argmax(Q2)]+alpha*np.max(Q))
print(np.mean(truth_1), np.mean(estimate_1))
print(np.mean(truth_1), np.mean(estimate_2))
print(np.mean(estimate_3))


