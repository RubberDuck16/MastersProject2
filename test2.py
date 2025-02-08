import numpy as np
from Utils.functions import *
import jax.numpy as jnp

k = 3
n_left = 3
n_right = 2
n = n_left * n_right

"""
V^T (A kron B) V

v: K x N x M
a: N x N
b: M x M
einsum(v,a,b,v, equation="k n m , n n', m m', k' n' m' -> k k' ")
"""

true_g = [
    initialise_g(n_left, n_right)
]

A = true_g[0]["left"]
B = true_g[0]["right"]
KP = np.kron(A, B)

key = jax.random.PRNGKey(0)
V = jax.random.normal(key, (k, n_left, n_right))

sketch_final = np.einsum('knm, ni, mj, fij -> kf', V, A, B, V)


V_resh = np.reshape(V, (k, n_left*n_right), order='C')      # we need to reshape since KP is 15 x 15 (PxP)

sketch_brute = np.einsum('ka, ab, fb -> kf', V_resh, KP, V_resh)

# is it possible to find V^T G V where G = KP, using only A and B?
# is it the same if we do V^T (A,B) V? -> yes!
# note: V will have to be reshaped in scenario 1 and 2 as G is a PxP matrix and A and B are NxN and MxM matrices respectively
print(sketch_final)
print(sketch_brute)

V = jax.random.normal(key, (n, k))
print('---------')
print(V.shape)

V_resh = np.reshape(V, (k, n_left, n_right), order='C')   
sketch_final = np.einsum('knm, ni, mj, fij -> kf', V_resh, A, B, V_resh)



# side note: order 'C' is numpys default so u can just leave it

V_resh2 = np.reshape(V, (k, n), order='C').T      # do this so V is reshaped in same way as it is for sketch final
# T so that it goes back to being nxk

#sketch_brute = np.einsum('ka, ab, fb -> kf', V_resh2, KP, V_resh2)


term1 = V_resh2.T @ KP @ V_resh2    # this is okay
term2 = V.T @ KP @ V                # this is not okay

print('---')
print(sketch_final)
print(sketch_brute)
print(term1)
print(term2)

print('-----')
print('V vs V_resh2')
print('V:')
print(V)
print('V_resh2:')
print(V_resh2)
print(np.reshape(np.reshape(V, (k, n_left, n_right), order='F'), (n, k), order='F'))

a = np.reshape(V, (k, n_left, n_right), order='F')
b = np.reshape(a, (k, n), order='F')
