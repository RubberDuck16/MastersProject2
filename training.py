import jax
import jax.numpy as jnp
import optax
from functions import *

# Parameters - P = n_left * n_right
k = 20
n_left = 100
n_right = 20


# Construct true_g
true_g = [
    initialise_g(n_left, n_right),
    initialise_g(n_left, n_right, alpha=0.1),
]


# Initialise learned_g (parameters we want to learn)
learned_g = [
    initialise_g(n_left, n_right),
]

# Loss function
def loss_fn(v, current_g):
    sketch_true = sketch(true_g, v)
    sketch_approx = sketch(current_g, v)
    return jnp.mean((sketch_approx - sketch_true) ** 2)


#### ADAM OPTIMISATION ####

# Optax optimiser configuration
learning_rate = 1e-4
beta2 = 0.99
optimizer = optax.adam(learning_rate=learning_rate, b2=beta2)

# Initialise optimiser state
params = learned_g
opt_state = optimizer.init(params)

# Parameter update step
@jax.jit
def update(params, opt_state, v):
    # there is a new v at every step so thats why its passed in
    
    # the loss used in jax.value_and_grad must be a function of the parameters only
    def compute_loss(params):
        return loss_fn(v, params)

    loss, grads = jax.value_and_grad(compute_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

losses = []

# Main loop
def train():
    global params, opt_state
    for t in range(300):
        key = jax.random.PRNGKey(t)
        v = jax.random.normal(key, shape=(k, n_left, n_right))
        params, opt_state, loss = update(params, opt_state, v)
        losses.append(loss)
        if t % 50 == 0:
            print(f"Iteration: {t}, Loss: {loss}")
        
        # Ensure loss.txt is empty before writing new data
        if t == 0:
            with open("loss.txt", "w") as f:
                f.write("")
                f.write(f"Title, Training Loss, k = {k}, n_left = {n_left}, n_right = {n_right}, P = {n_left*n_right}\n")

        with open("loss.txt", "a") as f:
            f.write(f"{t},{loss}\n")

train()

