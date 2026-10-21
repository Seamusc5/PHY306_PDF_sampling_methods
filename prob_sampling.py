import math
import random as rand
import numpy as np
import matplotlib.pyplot as plt

def dist(x):
    values = np.asarray(x)
    return np.where(values >= 0, np.exp(-np.maximum(values, 0)), 0.0)

def n_rej_samples(n, max_val):
    samples = []
    while len(samples) < n:
        x = rand.uniform(0, max_val)
        y = rand.uniform(0, 1)
        if y < dist(x):
            samples.append(x)
    return np.array(samples)

def inverse_transform_sampling(n, max_val=1):
    if max_val <= 0:
        raise ValueError("max_val must be greater than 0")

    samples = []
    for _ in range(n):
        u = rand.random()
        x = -math.log1p(-u * (1 - math.exp(-max_val)))
        samples.append(x)
    return np.array(samples)

def plot_histogram(samples, bins, max_val=1):
    plt.figure()
    plt.hist(samples, bins=bins, density=True, alpha=0.6, color='g', edgecolor='black')
    x = np.linspace(0, max_val, 100)
    pdf = dist(x) / (1 - math.exp(-max_val))
    plt.plot(x, pdf, 'r-', lw=2)
    plt.title('Histogram of Samples with PDF Overlay')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid()
    plt.show()

rejection = n_rej_samples(1000, 20)
inv_transform = inverse_transform_sampling(1000)
plot_histogram(rejection, 30, max_val=20)
plot_histogram(inv_transform, 30, max_val=1)
