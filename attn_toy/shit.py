import numpy as np

def f(n):
    term = 1
    logsum = 0
    for k in range(1,n):
        term *= n/k
        logsum+= term
    return np.exp(np.log(logsum)-n)

print(f(10),f(100),f(1000))