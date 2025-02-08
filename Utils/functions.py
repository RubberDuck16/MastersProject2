import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt


def random_pos_def_sqrt(n, key, alpha=1.0, eigs=None):
    """
    n is size of matrix
    key is a jax random key
    alpha is a scaling factor
    eigs is a list of eigenvalues (type = List)
    """
    q, _ = jnp.linalg.qr(jax.random.normal(key, shape=(n, n)))
    if eigs is None:
        s = alpha * jnp.exp(-jnp.arange(n) / (n / 4))   
    else:
        s = alpha * jnp.array(eigs)
    return q @ jnp.diag(jnp.sqrt(s))                                # so matrix is positive semi-definite


def initialise_g(n_left, n_right, key, alpha=1.0, eigs_left=None, eigs_right=None):
    """
    eigs_left and eigs_right are lists of eigenvalues for the left and right matrices respectively.
    """
    key_left, key_right = jax.random.split(key)         # split key so that we get a different random matrix for left and right
    return {
        "left": random_pos_def_sqrt(n_left, key_left, alpha, eigs_left),
        "right": random_pos_def_sqrt(n_right, key_right, alpha, eigs_right)
    }


def identity_guess(n_left, n_right):
    """
    Returns an initial guess of G as the identity matrix.
    """
    return {
        "left": jnp.eye(n_left),
        "right": jnp.eye(n_right)
    }


def vec(A):
    """
    Stacks the entries of matrix A column-wise to form a vector.
    A: Input matrix of shape (m, n)
    
    Returns:
    Vector of length m * n
    """
    return A.T.reshape(-1)


def row_vec(A):
    """
    Returns the row-major vectorization of A.
    Converts a m x n matrix to a mn x 1 vector.
    """
    return A.reshape(-1, order='C')


def sketch3(g_list, v):
    """
    Returns a K x K sketch.
    Assume that v is input as k x n_left x n_right.
    """
    full_sketch = 0.0
    for g in g_list:
        left = g["left"] @ g["left"].T
        right = g["right"] @ g["right"].T
        full_sketch += jnp.einsum('knm, ni, mj, fij -> kf', v, left, right, v)
    return full_sketch


def Gv_product(g_list, v):
    full_product = 0.0
    for g in g_list:
        left = g["left"] @ g["left"].T
        right = g["right"] @ g["right"].T
        full_product += jnp.einsum('ij,ab,kja->kib', left, right, v)
    return full_product


def initialise_v(n_left, n_right):
    left = np.random.randn(n_left)
    normed_left = left / np.linalg.norm(left)
    right = np.random.randn(n_right)
    normed_right = right / np.linalg.norm(right)
    return {
        "left": normed_left,
        "right": normed_right
    }


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
        v_flat = reshape(v, (-1, n_left * n_right)) # reshape v into (k, n_left*n_right) in order t multiply with gv
        
        full_sketch += v_flat @ gv.T
    return full_sketch 


def KP_sum(g_list):
    """
    Returns full matrix from a list of Kronecker products.
    """
    res = 0.0
    for g in g_list:
        left = g["left"] @ g["left"].T
        right = g["right"] @ g["right"].T

        KP = np.kron(left, right)
        res += KP
    return res


def optimise_G_hat(initial_guess, true_G, K=10, iters=20000, plot_losses=False):
    """
    Initial_guess: list of dictionaries containing left and right matrices, our initial apprxo. of G as just one KP in a list
    True_G: list of dictionaries containing left and right matrices, the true G is a sum of Kronecker products in a list
    K: sketching dimension
    Iters: number of iterations
    """
    n_left = initial_guess[0]["left"].shape[0]
    n_right = initial_guess[0]["right"].shape[0]

    optimizer = optax.adam(learning_rate=1e-4, b2=0.99)
    params = initial_guess
    opt_state = optimizer.init(params)

    def loss_fn(v, r, current_g):
        if type(true_G) is list:
            sketch_true = sketch3(true_G, v)
        else:
            # ground truth GGN is not a Kronecker product
            sketch_true = v.reshape(-1, n_left*n_right) @ true_G @ v.reshape(-1, n_left*n_right).T
        sketch_approx = r * sketch3(current_g, v)
        frobenius_norm = jnp.linalg.norm(sketch_true, 'fro')
        return jnp.mean((sketch_approx - sketch_true) ** 2) / frobenius_norm

    @jax.jit
    def update(params, opt_state, v, r):
        def compute_loss(params):
            return loss_fn(v, r, params)

        loss, grads = jax.value_and_grad(compute_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    losses = []
    for t in range(iters):
        key = jax.random.PRNGKey(t)
        v = jax.random.normal(key, shape=(K, n_left, n_right))
        
        # find r
        if t == 0:
            # find the scalar which we will multiply all subsequent guesses by to make them of the order of GGN
            if type(true_G) is list:
                sketch_true = sketch3(true_G, v)
            else:
                sketch_true = v.reshape(-1, n_left*n_right) @ true_G @ v.reshape(-1, n_left*n_right).T
            sketch_guess = sketch3(initial_guess, v)
            
            true_size = jnp.linalg.norm(sketch_true, 'fro')
            guess_size = jnp.linalg.norm(sketch_guess, 'fro')
            r = true_size / guess_size
            
        params, opt_state, loss = update(params, opt_state, v, r)
        losses.append(loss)
        if t % 50 == 0:
            print(f"Iteration: {t}, Loss: {loss}")
    
    if plot_losses:
        plt.plot(losses)
        plt.title("Training Loss when Approximating G as a Kronecker Product")
        plt.ylabel("Loss")
        plt.xlabel("Iteration")
        plt.show()
    return params, losses


def ground_truth_G(P, key=None, eigs=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    q, _ = jnp.linalg.qr(jax.random.normal(key, shape=(P, P)))
    if eigs is None:
        s = jnp.exp(-jnp.arange(P) / (P / 4))   
    else:
        s = jnp.array(eigs)
    return q.T @ jnp.diag(s) @ q


