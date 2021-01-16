import numpy as np
import time

np.random.seed(int(time.time()))
ep_len = 1000000
num_action = 100
snr = 0.5


def random_func(length):
    # return np.random.randn(length)
    # return np.random.rand(num_action)*2-1
    return np.random.beta(1, 1, num_action) - 1. / 2


x = np.random.rand(num_action)
print(np.max(x))
estimate = []
truth = []
for i in range(ep_len):
    Q = x + snr * random_func(num_action)
    estimate.append(np.max(Q))
    truth.append(x[np.argmax(Q)])

print(np.mean(truth), np.mean(estimate))

estimate = []
truth = []
for i in range(ep_len):
    Q1 = x + snr * random_func(num_action)
    Q2 = x + snr * random_func(num_action)
    estimate.append(Q1[np.argmax(Q2)])
    truth.append(x[np.argmax(Q1)])
    # estimate.append(min(np.max(Q1),np.max(Q2)))
    # estimate.append(Q1[np.random.randint(0,len(Q1))])

print(np.mean(truth), np.mean(estimate))
