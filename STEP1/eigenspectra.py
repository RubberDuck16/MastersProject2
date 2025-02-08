import numpy as np
import matplotlib.pyplot as plt
from Utils.algorithms import *

def GGN(eigenvalues):
    # Return a PxP curvature matrix from a list of P eigenvalues
    P = len(eigenvalues)
    u, _ = np.linalg.qr(np.random.randn(P, P)) 
    s = np.diag(np.array(eigenvalues)) 
    return u @ s @ u.T

def plot_eigenspectra(eigenvalue_options):
    for label, eigenvalues in eigenvalue_options.items():
        P = len(eigenvalues)
        plt.figure(figsize=(10, 6))
        plt.plot(range(P), eigenvalues, marker='o', linestyle='-', label=label)
        plt.title('Eigenvalue Spectrum (Log Scale)')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.grid(True)
        plt.legend()
        plt.show()

## Parameters
P = 200
K = 2    
iterations = 50  
N_trials = 50

eigenvalue_options = {
    "Uniform": [1 + 0.1 * np.random.uniform(-1, 1) for _ in range(P)],
    "Smooth Exponential Decay": list(np.exp(-np.arange(P) / (P / 4))),     
    "Exponential Decay": [5 * (0.9 ** i) for i in range(P)],
    "Dominant Eigenvalue": list(np.array([100] + [1] * (P - 1)))
}


#plot_eigenspectra(eigenvalue_options)

Ks = [2, 4, 20]
colors = ['b', 'g', 'm']

for label, eigenvalues in eigenvalue_options.items():
    sigma = GGN(eigenvalues)
    cond = np.linalg.cond(sigma)        # condition number of GGN

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    axs[0].plot(range(P), eigenvalues, marker='o', linestyle='-')
    axs[0].set_title(f'Eigenvalue Spectrum - {label} (κ = {cond:.2f})')
    axs[0].set_ylabel('Eigenvalue')  
    axs[0].grid(True)    

    for i, K in enumerate(Ks):
        ratio = K/P
        total_losses1 = np.zeros((N_trials, iterations))
        total_losses2 = np.zeros((N_trials, iterations))
        for trial in range(N_trials):
            print('Trial:', trial) 
            key = jax.random.PRNGKey(trial)
            total_losses1[trial][:], _ = original_SOFO_loss(P, K, sigma, key, iterations)
            #total_losses2[trial][:], _ = truncated_CT_SOFO_loss(P, K, sigma, key, iterations, truncation=10)
            total_losses2[trial][:], _ = CT_SOFO_loss(P, K, sigma, key, iterations)
        avg_losses1 = total_losses1.mean(axis=0)
        avg_losses2 = total_losses2.mean(axis=0)
        
        axs[1].plot(np.linspace(0, iterations, iterations), avg_losses1, linestyle='--', color=colors[i], label=f'Random search directions, {ratio*100}%')
        axs[1].plot(np.linspace(0, iterations, iterations), avg_losses2, color=colors[i], label=f'Conjugate search directions, {ratio*100}%')
        #axs[2].loglog(np.linspace(1, iterations, iterations-1), avg_losses1[1:], linestyle='--', color=colors[i], label=f'Random search directions, {ratio*100}%')
        #axs[2].loglog(np.linspace(1, iterations, iterations-1), avg_losses2[1:], color=colors[i], label=f'Conjugate search directions, {ratio*100}%')


    axs[1].set_title(f'Average Training Loss across {N_trials} Trials')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    #axs[2].set_title(f'Average Log Training Loss across {N_trials} Trials')
    #axs[2].set_xlabel('Iteration')
    #axs[2].set_ylabel('Loss')
    #axs[2].legend()

    plt.constrained_layout=True
    plt.savefig(f'Plots/{label}_losses.png')
    plt.show()


