import numpy as np

def matrix_transpose(A )-> np.ndarray:
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    # Write code here
    res = []
    for i in range (A.shape[1]):
        res.append(A[:,i].flatten())
    return np.array(res) 
