import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    W_hh_norm = np.linalg.norm(W_hh, ord=2)
    gradient = []
    current_gradient = 1.0 
    for i in range ( T ) : 
        gradient.append ( current_gradient)
        current_gradient = current_gradient*W_hh_norm
    return gradient 