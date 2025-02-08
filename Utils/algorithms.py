import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from collections import deque
from Utils.functions import *


def GGN(eigenvalues):
    # Return a PxP curvature matrix from a list of P eigenvalues
    P = len(eigenvalues)
    u, _ = np.linalg.qr(np.random.randn(P, P)) 
    s = np.diag(np.array(eigenvalues)) 
    return u @ s @ u.T


def original_SOFO_loss(P, K, sigma, key, N, learning_rate=1):
    losses = []
    condition_number = []
    theta_min = np.zeros(P)
    current_theta = jax.random.normal(key, shape=(P,))
    loss = 0.5 * (current_theta-theta_min).T @ sigma @ (current_theta-theta_min)
    losses.append(loss)

    for _ in range(N-1):
        v, _ = np.linalg.qr(np.random.randn(P, P))  
        v = v[:, :K]   

        c = v.T @ sigma @ v 
        #condition_number.append(np.linalg.cond(c))
        g = v.T @ sigma @ (current_theta-theta_min)   
        dtheta = v @ np.linalg.inv(c) @ g          
        current_theta = current_theta - learning_rate * dtheta  

        loss = 0.5 * (current_theta-theta_min).T @ sigma @ (current_theta-theta_min)
        losses.append(loss)
    return losses, condition_number


def CT_SOFO_loss(P, K, sigma, key, N, learning_rate=1):
    losses = []
    condition_number = []
    
    theta_min = np.zeros(P)
    current_theta = jax.random.normal(key, shape=(P,))
    loss = 0.5 * (current_theta-theta_min).T @ sigma @ (current_theta-theta_min)
    losses.append(loss)

    previous_vs = {}   
    for _ in range(N-1):
        V_list = []                     # holds conjugate search directions for each iteration

        v_orthog, _ = np.linalg.qr(np.random.randn(P, P))  
        for i in range(1, K+1):
            v = v_orthog[:, i-1]

            if i in previous_vs.keys(): 
                for j in range(len(previous_vs[i])):        # check all previous search directions in this K dimension
                    vj = previous_vs[i][j]
                    num = v.T @ sigma @ vj
                    denom = vj.T @ sigma @ vj
                    v -= (num/denom) * vj
                
                previous_vs[i].append(v)
                
                for j in range(len(previous_vs[i]) - 1):  
                    print(f"Conjugacy check with vector {j}: {v.T @ sigma @ previous_vs[i][j]}")
            else:
                previous_vs[i] = [v]            # for K dimension i, add the first search direction
            
            V_list.append(v)                # add conjugated v to the list that will hold the K v's
        V = np.column_stack(V_list)         # stack the K v's to form the V matrix

        c = V.T @ sigma @ V
        #condition_number.append(np.linalg.cond(c))
        g = V.T @ sigma @ (current_theta-theta_min)   
        dtheta = V @ np.linalg.inv(c) @ g          
        current_theta = current_theta - learning_rate * dtheta   

        loss = 0.5 * (current_theta-theta_min).T @ sigma @ (current_theta-theta_min)
        losses.append(loss)
    return losses, condition_number


def truncated_CT_SOFO_loss(P, K, sigma, key, N, truncation, learning_rate=1):
    losses = []
    condition_number = []
    
    theta_min = np.zeros(P)
    current_theta = jax.random.normal(key, shape=(P,))
    loss = 0.5 * (current_theta-theta_min).T @ sigma @ (current_theta-theta_min)
    losses.append(loss)

    previous_vs = {i: deque(maxlen=truncation) for i in range(1, K+1)}  
    for _ in range(N-1):
        V_list = []                # holds conjugate search directions for each iteration

        v_orthog, _ = np.linalg.qr(np.random.randn(P, P))   
        for a in range(1, K+1): 
            v = v_orthog[:, a-1]     

            if previous_vs[a]:
                for vj in previous_vs[a]:
                    num = v.T @ sigma @ vj
                    denom = vj.T @ sigma @ vj
                    v -= (num / denom) * vj

            previous_vs[a].append(v)
            V_list.append(v)
        
        V = np.column_stack(V_list)

        c = V.T @ sigma @ V
        condition_number.append(np.linalg.cond(c))
        g = V.T @ sigma @ (current_theta-theta_min)   
        dtheta = V @ np.linalg.inv(c) @ g          
        current_theta = current_theta - learning_rate * dtheta     

        # loss on each iteration
        loss = 0.5 * (current_theta-theta_min).T @ sigma @ (current_theta-theta_min)
        losses.append(loss)
    return losses, condition_number


def KP_truncated_CT_SOFO_loss(P, K, sigma, key, N, G_approx, damping_factor=2, learning_rate=1):
    G_l = G_approx[0]["left"] @ G_approx[0]["left"].T
    G_r = G_approx[0]["right"] @ G_approx[0]["right"].T
    n_left = G_l.shape[0]
    n_right = G_r.shape[0]
    
    losses, condition_number = [], []
    theta_min = np.zeros(P)
    current_theta = jax.random.normal(key, shape=(P,))
    loss = 0.5 * (current_theta-theta_min).T @ sigma @ (current_theta-theta_min)
    losses.append(loss)

    truncation = min(n_left, n_right)

    previous_vs = {i: deque(maxlen=truncation) for i in range(1, K+1)}  
    for n in range(N-1):
        V_list = []                # holds conjugate search directions for each iteration

        v_orthog1, _ = np.linalg.qr(np.random.randn(n_left, P))         # also ensures that vs are normalised so no numerical issues
        v_orthog2, _ = np.linalg.qr(np.random.randn(n_right, P))
        for k in range(1, K+1):
            v_l = v_orthog1[:, k-1]
            v_r = v_orthog2[:, k-1]

            if previous_vs[k]:
                for vj in previous_vs[k]:
                    vj_l = vj["left"]
                    vj_r = vj["right"]
                    
                    num = v_l.T @ G_l @ vj_l
                    denom = vj_l.T @ G_l @ vj_l
                    v_l -= (num / denom) * vj_l

                    num = v_r.T @ G_r @ vj_r
                    denom = vj_r.T @ G_r @ vj_r
                    v_r -= (num / denom) * vj_r

                kp = np.kron(v_l, v_r)
                print('Iteration:', n)  
                print('Conjugacy check with previous tangent: ', kp @ sigma @ np.kron(previous_vs[k][-1]["left"], previous_vs[k][-1]["right"]))

            v = {"left": v_l, "right": v_r}
            
            previous_vs[k].append(v)
            V_list.append(np.kron(v_l, v_r)) 

        
        V = np.column_stack(V_list)
        #Q, _ = np.linalg.qr(V)
        #V = Q
        # canhave V_l and V_r which contain the k left vectors and k right vectors respectively
        
        c = V.T @ sigma @ V
        
        # compute SVD of sketch
        #_, s, _ = jnp.linalg.svd(c)
        #s_max = jnp.max(jnp.diag(s))

        #new_c = c + (s_max*damping_factor)*jnp.eye(c.shape[0])
        #condition_number.append(np.linalg.cond(c))

        g = V.T @ sigma @ (current_theta-theta_min)   
        dtheta = V @ np.linalg.inv(c) @ g          
        current_theta = current_theta - learning_rate * dtheta     

        # loss on each iteration
        loss = 0.5 * (current_theta-theta_min).T @ sigma @ (current_theta-theta_min)
        losses.append(loss)
    return losses, condition_number


