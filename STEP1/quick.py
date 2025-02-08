import matplotlib.pyplot as plt

P = 200

eigs = [(0.5 ** i) for i in range(P)]


plt.plot(range(P), eigs, marker='o', linestyle='-', color='black')
plt.title(f'Eigenvalue Spectrum')
plt.ylabel('Eigenvalue')  
plt.grid(True)    
plt.show()
