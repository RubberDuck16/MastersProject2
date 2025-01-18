import jax
import jax.numpy as jnp
import numpy as np
import optax


def random_pos_def_sqrt(n, alpha=1.0):
    key = jax.random.PRNGKey(0)
    q, _ = jnp.linalg.qr(jax.random.normal(key, shape=(n, n)))
    s = alpha * jnp.exp(-jnp.arange(n) / (n / 4))
    return q @ jnp.diag(jnp.sqrt(s))


def initialise_g(n_left, n_right, alpha=1.0):
    return {
        "left": random_pos_def_sqrt(n_left, alpha),
        "right": random_pos_def_sqrt(n_right),
    }

def vec(A):
    """
    Stacks the entries of matrix A column-wise to form a vector.
    A: Input matrix of shape (m, n)
    
    Returns:
    Vector of length m * n
    """
    return A.T.reshape(-1)


# Sketch function
def sketch1(g_list, v):
    """
    Returns a K x K sketch.
    """
    full_sketch = 0.0
    for g in g_list:
        n_left = g["left"].shape[0]
        n_right = g["right"].shape[0]

        left = g["left"] @ g["left"].T
        right = g["right"] @ g["right"].T
        gv = np.einsum('ij,ab,kja->kib', left, right, v)
                 
        gv = gv.reshape(-1, n_left * n_right)                # make both the same shape for multiplicaiton
        v_flat = v.reshape(-1, n_left * n_right)

        full_sketch += v_flat @ gv.T
    return full_sketch




def reshape(v, shape):
    return v.reshape(shape)

# This is the sketch function with all reshaped laid out (no einsum)
def sketch(g_list, v):   
    """
    v is a k x n_left x n_right tensor
    """
    full_sketch = 0.0
    for g in g_list:
        n_left = g["left"].shape[0]
        n_right = g["right"].shape[0]

        left = g["left"] @ g["left"].T
        right = g["right"] @ g["right"].T
      
        gv = (reshape(v, (-1, n_right)) @ right)    # reshape v into (k*n_left, n_right) and multiply by right
        gv = reshape(gv, (-1, n_left, n_right))     # reshape gv into (k, n_left, n_right)
        
        gv = gv.transpose(0, 2, 1)                  # transpose gv to (k, n_right, n_left)
        gv = (reshape(gv, (-1, n_left)) @ left)     # reshape gv into (k*n_right, n_left) and multiply by left
        gv = reshape(gv, (-1, n_right, n_left))     # reshape gv into (k, n_right, n_left)
        gv = gv.transpose(0, 2, 1)                  # transpose gv back to (k, n_left, n_right)

        gv = reshape(gv, (-1, n_left * n_right))    # reshape gv into (k, n_left*n_right)
        v_flat = reshape(v, (-1, n_left * n_right))     # reshape v into (k, n_left*n_right) in order t multiply with gv
        
        full_sketch += v_flat @ gv.T
    return full_sketch 



def multiplication(g_list, a):   
    """
    a is a P x 1 tensor
    """
    a = reshape(a, (n_left, n_right))
    full_multiplcation = 0.0
    for g in g_list:
        n_left = g["left"].shape[0]
        n_right = g["right"].shape[0]

        left = g["left"] @ g["left"].T
        right = g["right"] @ g["right"].T

        ga = (reshape(a, (n_left, n_right)) @ right)        # n_left x n_right
        ga = ga.transpose(1, 0)                             # n_right x n_left
        ga = ga @ left                                      # n_right x n_left
        ga = reshape(ga, (n_left*n_right, 1))               # n_left*n_right x 1

        full_multiplcation += ga
    return full_multiplcation 


def multiplication2(g_list, a):
    full_multiplcation = 0.0

    # a is the vectorisation of x 
    for g in g_list:
        n_left = g["left"].shape[0]
        n_right = g["right"].shape[0]

        left = g["left"] @ g["left"].T
        right = g["right"] @ g["right"].T

        a_reshaped = reshape(a, (n_left, n_right))

        term = left @ (a_reshaped @ right)
        term = vec(term)
        full_multiplcation += term
    return full_multiplcation



def curvature_matrix(g_list):
    res = 0.0
    for g in g_list:
        left = g["left"] @ g["left"].T
        right = g["right"] @ g["right"].T

        KP = np.kron(left, right)
        res += KP
    return res



# stop after a certain number of iterations and return the approximated G
def approximated_G(N, n_left, n_right, k, true_g):
    def loss_fn(v, current_g):
        sketch_true = sketch(true_g, v)
        sketch_approx = sketch(current_g, v)
        return jnp.mean((sketch_approx - sketch_true) ** 2)
    
    @jax.jit
    def update(params, opt_state, v):
        def compute_loss(params):
            return loss_fn(v, params)

        loss, grads = jax.value_and_grad(compute_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    learned_g = [initialise_g(n_left, n_right)]
    
    # ADAM optimiser
    learning_rate = 1e-4
    beta2 = 0.99
    params = learned_g
    optimizer = optax.adam(learning_rate=learning_rate, b2=beta2)
    opt_state = optimizer.init(params)

    for t in range(N):
        key = jax.random.PRNGKey(t)
        v = jax.random.normal(key, shape=(k, n_left, n_right))
        params, opt_state, loss = update(params, opt_state, v)
    
    # params will be a list with just one entry - the dictionary that we want
    return params


# params_estimated["left"] is left matrix in KP
# params_estimated["right"] is right matrix in KP


