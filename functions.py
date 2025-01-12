import numpy as np


def curvature_matrix(n, decay_factor=0.5):
    """
    Make the ground truth curvature matrix.
    
    n: int, the dimension of the curvature matrix.
    decay_factor: float, the decay factor of the eigenvalues (how ill-conditioned you want curvature matrix).
    """
    u, _ = np.linalg.qr(np.random.randn(n, n))        # QR decomposition of a Gaussian matrix
    
    initial_value = 50
    eigenvalues = [initial_value*(decay_factor**i) for i in range(n)]
    s = np.diag(np.array(eigenvalues))                                     
    return u @ s @ u.T



