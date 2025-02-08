import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from collections import deque
from Utils.algorithms import *



# i think what happens is u conjugate so much that the vectors start to 
# become linearly dependent and the condition number blows up


## Parameters
P = 200
K = 2    
iterations = 50              # training iterations

eigenvalues = [10 * (0.5 ** i) for i in range(P)]
sigma = GGN(eigenvalues)



N_trials = 10
total_losses1 = np.zeros((N_trials, iterations))
total_losses2 = np.zeros((N_trials, iterations))
cn1 = np.zeros((N_trials, iterations-1))
cn2 = np.zeros((N_trials, iterations-1))

for trial in range(N_trials):
    key = jax.random.PRNGKey(trial)
    total_losses1[trial][:], cn1[trial][:] = original_SOFO_loss(P, K, sigma, key, iterations)
    total_losses2[trial][:], cn2[trial][:] = truncated_CT_SOFO_loss(P, K, sigma, key, iterations, truncation=10)
    #total_losses2[trial][:], cn2[trial][:] = CT_SOFO_loss(P, K, sigma, key, iterations)
avg_losses1 = total_losses1.mean(axis=0)
avg_losses2 = total_losses2.mean(axis=0)
cn1 = cn1.mean(axis=0)
cn2 = cn2.mean(axis=0)


fig, axs = plt.subplots(3, 1, figsize=(8, 13))

# Plot 1: Average Training Loss
axs[0].plot(np.linspace(0, iterations, iterations), avg_losses1, label='Random search directions')
axs[0].plot(np.linspace(0, iterations, iterations), avg_losses2, label='Conjugate search directions')
axs[0].set_title(f'Average Training Loss across {N_trials} Trials')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Plot 2: Log log plot
axs[1].loglog(np.linspace(1, iterations, iterations-1), avg_losses1[1:], label='Random search directions')
axs[1].loglog(np.linspace(1, iterations, iterations-1), avg_losses2[1:], label='Conjugate search directions')
axs[1].set_title(f'Average Log Training Loss across {N_trials} Trials')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Loss')
axs[1].legend()

# Plot 3: Average Sketched GGN Condition Number
axs[2].plot(np.linspace(0, iterations-1, iterations-1), cn1, label='Random search directions', linestyle='dashed')
axs[2].plot(np.linspace(0, iterations-1, iterations-1), cn2, label='Conjugate search directions', linestyle='dashed')
axs[2].set_title(f'Average Sketched GGN Condition Number across {N_trials} Trials')
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Condition Number')
axs[2].legend()

plt.tight_layout()
plt.show()



