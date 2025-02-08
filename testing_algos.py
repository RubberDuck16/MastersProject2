import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from Utils.functions import *
from Utils.algorithms import *

# Parameters
n_left = 27
n_right = 13
P = n_left * n_right

eigs_left = [(0.5 ** i) for i in range(n_left)]
eigs_right = [(0.5 ** i) for i in range(n_right)]

G = [
    initialise_g(n_left, n_right, jax.random.PRNGKey(40), eigs_left=eigs_left, eigs_right=eigs_right)
]
sigma = KP_sum(G)

fig, axs = plt.subplots(2, 2, figsize=(15, 5))

eigenvalues, _ = jnp.linalg.eigh(sigma)
axs[0][0].plot(eigenvalues[::-1], marker='o', linestyle='-', color='black')
axs[0][0].set_title(f'Eigenvalue Spectrum of Ground Truth G')
axs[0][0].set_ylabel('Eigenvalue')  
axs[0][0].grid(alpha=0.5)    


key = jax.random.PRNGKey(7)     # initial parameter guess key
K = 1
N = 500

sofo_loss, sofo_condition = original_SOFO_loss(P, K, sigma, key, N)
print('Done SOFO loss')
truncated_CT_loss, truncated_CT_condition = truncated_CT_SOFO_loss(P, K, sigma, key, N, truncation=min(n_left, n_right))
print('Done truncated CT loss')
truncated_CT_loss_KP, CT_KP_condition = KP_truncated_CT_SOFO_loss(P, K, sigma, key, N, G_approx=G)


axs[1][0].plot(np.linspace(0, N, N), sofo_loss, alpha=0.7, label=f'SOFO')
axs[1][0].plot(np.linspace(0, N, N), truncated_CT_loss, alpha=0.7, label=f'Full G conjugate tangents')
axs[1][0].plot(np.linspace(0, N, N), truncated_CT_loss_KP, alpha=0.7, label=f'KP conjugate tangents')
axs[1][0].legend()
axs[1][0].set_title('Training Loss')
axs[1][0].set_xlabel('Iteration')
axs[1][0].grid(alpha=0.5)

axs[1][1].loglog(np.linspace(1, N, N-1), sofo_loss[1:], alpha=0.7, label=f'SOFO')
axs[1][1].loglog(np.linspace(1, N, N-1), truncated_CT_loss[1:], alpha=0.7, label=f'Full G conjugate tangents')
axs[1][1].loglog(np.linspace(1, N, N-1), truncated_CT_loss_KP[1:], alpha=0.7, label=f'KP conjugate tangents')
axs[1][1].legend()
axs[1][1].set_title('Log Training Loss')
axs[1][1].set_xlabel('Iteration')
axs[1][1].grid(alpha=0.5)

axs[0][1].plot(np.linspace(1, N, N-1), sofo_condition, alpha=0.7, label=f'SOFO')
axs[0][1].plot(np.linspace(1, N, N-1), truncated_CT_condition, alpha=0.7, label=f'Full G conjugate tangents')
axs[0][1].plot(np.linspace(1, N, N-1), CT_KP_condition, alpha=0.7, label=f'KP conjugate tangents')
axs[0][1].legend()
axs[0][1].set_title('Condition Number')
axs[0][1].set_xlabel('Iteration')
axs[0][1].grid(alpha=0.5)

plt.constrained_layout=True
plt.show()