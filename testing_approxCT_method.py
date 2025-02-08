import jax
import jax.numpy as jnp
from Utils.functions import *
import matplotlib.pyplot as plt
from Utils.algorithms import *

# Parameters
n_left = 27
n_right = 13
P = n_left * n_right

eigs_left = 10*np.exp(-jnp.arange(n_left) / (n_left / 4))
eigs_right = 10*np.exp(-jnp.arange(n_right) / (n_right / 4))

#eigs_left = [10*(0.5 ** i) for i in range(n_left)]
#eigs_right = [10*(0.5 ** i) for i in range(n_right)]

# Ground truth G
G = [
    initialise_g(n_left, n_right, jax.random.PRNGKey(1), eigs_left=eigs_left, eigs_right=eigs_right)
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


truncation = 13                 # should be the smaller of n_left and n_right
K = 4
N = 1500
N_trials = 2

colours5 = ['red', 'blue', 'green']

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

keys2 = [jax.random.PRNGKey(x) for x in [2, 16, 23, 27, 35, 40, 55, 58, 80, 103]]
keys3 = [jax.random.PRNGKey(x) for x in [5, 21, 32, 34, 41, 47, 68, 70, 79, 96]]
keys4 = [jax.random.PRNGKey(x) for x in [8, 11, 15, 29, 30, 45, 51, 63, 69, 101]]
N_trials = len(keys2)

total_losses1 = np.zeros((N_trials, N))
total_losses2 = np.zeros((N_trials, N))
for trial in range(N_trials):
    print('Trial:', trial) 
    key = jax.random.PRNGKey(trial)                 # this changes the starting guess of parameters
    
    G = [initialise_g(n_left, n_right, keys2[trial], eigs_left=eigs_left, eigs_right=eigs_right),
        initialise_g(n_left, n_right, keys3[trial], alpha=0.1, eigs_left=eigs_left, eigs_right=eigs_right),
        initialise_g(n_left, n_right, keys3[trial], alpha=0.05, eigs_left=eigs_left, eigs_right=eigs_right)
    ]
    sigma = KP_sum(G)
    
    total_losses1[trial][:], _ = original_SOFO_loss(P, K, sigma, key, N)
    total_losses2[trial][:], _ = truncated_CT_SOFO_loss(P, K, sigma, key, N, truncation)
avg_losses1 = total_losses1.mean(axis=0)
avg_losses2 = total_losses2.mean(axis=0)
axs[1].loglog(np.linspace(1, N, N-1), avg_losses1[1:], linestyle='--', label=f'Random tangents')
axs[1].loglog(np.linspace(1, N, N-1), avg_losses2[1:], linestyle='--', label=f'Truncated conjugate tangents')


iters = [1000, 7500, 15000]
for i, iter in enumerate(iters):
    total_losses3 = np.zeros((N_trials, N))
    for trial in range(N_trials):
        print('Trial:', trial) 
        key = jax.random.PRNGKey(trial)

        G = [
            initialise_g(n_left, n_right, keys2[trial], eigs_left=eigs_left, eigs_right=eigs_right),
            initialise_g(n_left, n_right, keys3[trial], alpha=0.1, eigs_left=eigs_left, eigs_right=eigs_right),
            initialise_g(n_left, n_right, keys3[trial], alpha=0.05, eigs_left=eigs_left, eigs_right=eigs_right)
        ]
        sigma = KP_sum(G)
        
        G_approx, loss_adam = optimise_G_hat(initial_guess, G, K=5, iters=iter)
        total_losses3[trial][:] = KP_truncated_CT_SOFO_loss(P, K, sigma, key, N, truncation, G_approx)
 
    avg_losses3 = total_losses3.mean(axis=0)
       
    axs[1].loglog(np.linspace(1, N, N-1), avg_losses3[1:], color=colours5[i], label=f'Approximate conjugate tangents ({iter} iters)')


axs[1].legend()
axs[1].set_title(f'Average Log Training Loss across {N_trials} Trials')
axs[1].set_xlabel('SOFO Iteration')
axs[1].set_ylabel('Log Loss')
axs[1].grid(alpha=0.5)


axs[0].plot(loss_adam)
axs[0].set_title('Learning Approximation of G')
for i, iter in enumerate(iters):
    axs[0].axvline(x=iter, linestyle='--', color=colours5[i])
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Iteration')
axs[0].grid(alpha=0.5)

plt.constrained_layout=True
plt.show()
