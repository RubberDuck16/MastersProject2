import numpy as np
import jax.numpy as jnp
from Utils.functions import *

n_left = 5
n_right = 3
K = 2
P = n_left * n_right


initial_guess = [
    initialise_g(n_left, n_right, jax.random.PRNGKey(3))
]

intial_guess2 = [
    identity_guess(n_left, n_right)
]


          

n = n_left
eigenvalues_l = [
    10 * jnp.exp(-jnp.arange(n) / (n / 4)),
    10 * jnp.exp(-jnp.arange(n) / (n / 8)),
    jnp.array([10 * (0.5 ** i) for i in range(n)])
]

n = n_right
eigenvalues_r = [
    10 * jnp.exp(-jnp.arange(n) / (n / 4)),
    10 * jnp.exp(-jnp.arange(n) / (n / 8)),
    jnp.array([10 * (0.5 ** i) for i in range(n)])
]

true_G = [initialise_g(n_left, n_right, jax.random.PRNGKey(1), eigs_left=eigenvalues_l[0], eigs_right=eigenvalues_r[0])]

left = true_G[0]["left"] @ true_G[0]["left"].T
left_size = jnp.linalg.norm(left, 'fro')

right = true_G[0]["right"] @ true_G[0]["right"].T
right_size = jnp.linalg.norm(right, 'fro')

intial_guess2[0]["right"] *= right_size
intial_guess2[0]["left"] *= left_size

print(right)
print(right_size)

print(initial_guess[0]["right"])
print(intial_guess2[0]["right"])

print(left_size)
print(jnp.linalg.norm(intial_guess2[0]["left"], 'fro'))