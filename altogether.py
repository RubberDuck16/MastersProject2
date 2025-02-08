import jax
import jax.numpy as jnp
from Utils.functions import *
import matplotlib.pyplot as plt
from Utils.algorithms import *

# Parameters
n_left = 27
n_right = 13
P = n_left * n_right

eigs_left = np.exp(-jnp.arange(n_left) / (n_left / 4))
eigs_right = np.exp(-jnp.arange(n_right) / (n_right / 4))

eigs_left = [10*(0.5 ** i) for i in range(n_left)]
eigs_right = [10*(0.5 ** i) for i in range(n_right)]

# Ground truth G
G = [
    initialise_g(n_left, n_right, jax.random.PRNGKey(40), eigs_left=eigs_left, eigs_right=eigs_right)
]

sigma = KP_sum(G)

eigenvalues, _ = jnp.linalg.eigh(sigma)
eigenvalues = eigenvalues[::-1]

left_eigenvalues, _ = jnp.linalg.eigh(G[0]["left"] @ G[0]["left"].T)
left_eigenvalues = left_eigenvalues[::-1]

right_eigenvalues, _ = jnp.linalg.eigh(G[0]["right"] @ G[0]["right"].T)
right_eigenvalues = right_eigenvalues[::-1]

# Initial guess for approximation of G
initial_guess = [
    initialise_g(n_left, n_right, jax.random.PRNGKey(3))
]




plt.plot(left_eigenvalues)
plt.title('Left eigenvalues')
plt.show()
plt.plot(right_eigenvalues)
plt.title('Right eigenvalues')
plt.show()




plt.plot(eigenvalues, marker='o', linestyle='-', color='black')
plt.title(f'Eigenvalue Spectrum of Ground Truth G')
plt.ylabel('Eigenvalue')  
plt.grid(alpha=0.5)    
plt.show()


key = jax.random.PRNGKey(7)
truncation = 13                 # should be the smaller of n_left and n_right
K = 4
N = 1500

iters = [1000, 7500, 15000]
iters = [100]

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

colours5 = ['red', 'blue', 'green']


sofo_loss, _ = original_SOFO_loss(P, K, sigma, key, N)
losses3, _ = truncated_CT_SOFO_loss(P, K, sigma, key, N, truncation)

for i, iter in enumerate(iters):
    G_approx, loss_adam = optimise_G_hat(initial_guess, G, K=5, iters=iter)

    losses = KP_truncated_CT_SOFO_loss(P, K, sigma, key, N, truncation, G_approx=G)
    axs[1].loglog(np.linspace(1, N, N-1), losses[1:], color=colours5[i], label=f'Approximate conjugate tangents ({iter} iters)')
    print(losses[-1])

axs[1].loglog(np.linspace(1, N, N-1), sofo_loss[1:], linestyle='--', label=f'Random tangents')
axs[1].loglog(np.linspace(1, N, N-1), losses3[1:], linestyle='--', label=f'Truncated conjugate tangents')
axs[1].legend()
axs[1].set_title('Average Log Training Loss across __ Trials')
axs[1].set_xlabel('SOFO Iteration')
axs[1].set_ylabel('Log Loss')

axs[0].plot(loss_adam)
axs[0].set_title('Learning Approximation of G')
for i, iter in enumerate(iters):
    axs[0].axvline(x=iter, linestyle='--', color=colours5[i])
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Iteration')

plt.constrained_layout=True
plt.show()

