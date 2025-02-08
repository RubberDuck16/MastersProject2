import numpy as np
import matplotlib.pyplot as plt
from Utils.algorithms import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


my_colours = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']

## Parameters
P = 200
K = 2    
iterations = 100  
N_trials = 100

ratio = K/P

eigenvalue_options = {
    "Uniform": [1 + 0.1 * np.random.uniform(-1, 1) for _ in range(P)],
    "Smooth Exponential Decay": list(np.exp(-np.arange(P) / (P / 4))),     
    "Exponential Decay": [(0.5 ** i) for i in range(P)],
    "Dominant Eigenvalue": list(np.array([100] + [1] * (P - 1)))
}



sigma = GGN(eigenvalue_options["Exponential Decay"])

truncations = [1, 5, 10]

fig, axs = plt.subplots(2, 3, figsize=(15, (15/3)*2))

for i, truncation in enumerate(truncations):
    total_losses1 = np.zeros((N_trials, iterations))
    total_losses2 = np.zeros((N_trials, iterations))
    for trial in range(N_trials):
        print('Trial:', trial) 
        key = jax.random.PRNGKey(trial)
        total_losses1[trial][:], _ = original_SOFO_loss(P, K, sigma, key, iterations)
        total_losses2[trial][:], _ = truncated_CT_SOFO_loss(P, K, sigma, key, iterations, truncation)
    avg_losses1 = total_losses1.mean(axis=0)
    avg_losses2 = total_losses2.mean(axis=0)
        
    axs[0][i].plot(np.linspace(0, iterations, iterations), avg_losses1, color='grey', linestyle='--', alpha=0.7, label=f'Random tangents')
    axs[0][i].plot(np.linspace(0, iterations, iterations), avg_losses2, color='royalblue', label=f'Conjugate tangents (CT)')
    axs[1][i].loglog(np.linspace(1, iterations, iterations-1), avg_losses1[1:], color='grey', linestyle='--', alpha=0.7)
    axs[1][i].loglog(np.linspace(1, iterations, iterations-1), avg_losses2[1:], color='royalblue')

    axs[1][i].set_xlabel('Iteration')
    axs[1][i].grid(alpha=0.5)
    axs[1][i].set_ylabel('Log Loss')

    axs[0][i].set_title(f'Previous {truncation} Tangents')
    axs[0][i].set_ylabel('Loss')
    axs[0][i].legend()
    axs[0][i].grid(alpha=0.5)
    majorLocator = MultipleLocator(25)
    axs[0][i].xaxis.set_major_locator(majorLocator)
    

plt.suptitle(f'Average Training Loss aacross {N_trials} Trials, K/P = {int(ratio*100)}%')
plt.constrained_layout=True
plt.savefig(f'Plots/truncation_losses.png')
plt.show()



