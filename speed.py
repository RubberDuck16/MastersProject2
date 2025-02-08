import numpy as np 
from Utils.algorithms import *
import jax
import time
from Utils.functions import *

# test the speed of the algorithms

n_left = 27
n_right = 13
P = n_left * n_right

truncation = 13              
K = 4
N = 500


eigs_left = 10*np.exp(-jnp.arange(n_left) / (n_left / 4))
eigs_right = 10*np.exp(-jnp.arange(n_right) / (n_right / 4))



initial_guess = [
    initialise_g(n_left, n_right, jax.random.PRNGKey(3))
]


SOFO_times = []
CT_times = []
approx_CT_times = []

for trial in range(10):
    print(trial)
    G = [
    initialise_g(n_left, n_right, jax.random.PRNGKey(trial), eigs_left=eigs_left, eigs_right=eigs_right)
    ]
    sigma = KP_sum(G)
    G_approx, _ = optimise_G_hat(initial_guess, G, K=5, iters=10000)

    start_time = time.time()
    sofo_losses, _ = original_SOFO_loss(P, K, sigma, jax.random.PRNGKey(trial+5), N)
    sofo_time = time.time() - start_time
    SOFO_times.append(sofo_time)

    start_time = time.time()
    truncated_CT_losses, _ = truncated_CT_SOFO_loss(P, K, sigma, jax.random.PRNGKey(trial+5), N, truncation)
    truncated_CT_time = time.time() - start_time
    CT_times.append(truncated_CT_time)

    start_time = time.time()
    approximate_CT_losses = KP_truncated_CT_SOFO_loss(P, K, sigma, jax.random.PRNGKey(trial+5), N, truncation, G_approx)
    approximate_CT_time = time.time() - start_time
    approx_CT_times.append(approximate_CT_time)

print('Averaged over 50 trials')
print(f"Original SOFO loss time: {np.mean(SOFO_times):.3f} +/- {np.var(SOFO_times):.3f} seconds")
print(f"Truncated CT SOFO loss time: {np.mean(CT_times):.3f} +/- {np.var(CT_times):.3f} seconds")
print(f"Approximate CT SOFO loss time: {np.mean(approx_CT_times):.3f} +/- {np.var(approx_CT_times):.3f} seconds")