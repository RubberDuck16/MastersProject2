import numpy as np
from Utils.functions import *
from numpy.linalg import inv, qr
import matplotlib.pyplot as plt

# Parameters
k = 2
n_left = 5
n_right = 3
n = n_left * n_right

true_g = [
    initialise_g(n_left, n_right)
]

V, _ = qr(np.random.randn(n, n))
V = V[:, :k]  

### Original sketch
sigma = curvature_matrix(true_g)
c = V.T @ sigma @ V
print(c)


V_reshaped = V.reshape(k, n_left, n_right)

### Sketching using reshaping method
c2 = sketch(true_g, V_reshaped)
print(c2) 

### Sketching using einsum method 
c3 = sketch1(true_g, V_reshaped)
print(c3)



C = np.zeros((k, k))
for i in range(k):
    print('\n')
    v = V[:, i]
    print(v.shape)
    v_reshaped = v.reshape(1, n_left, n_right)
    c = v.T @ sigma @ v
    print(c)
    C[:, i] = c

    c2 = sketch(true_g, v_reshaped)
    print(c2) 

    c3 = sketch1(true_g, v_reshaped)
    print(c3)



print('---')
print(C)