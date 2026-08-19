import numpy as np

def candidate_hidden(h_prev: np.ndarray, x_t: np.ndarray, r_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Compute candidate: h_tilde = tanh(W_h @ [r*h, x] + b_h)
    """
    h_prev = np.asarray ( h_prev)
    x_t =  np.asarray ( x_t )
    r_t = np.asarray ( r_t)
    W_h = np.asarray ( W_h)
    b_h = np.asarray ( b_h)
    was_1d = h_prev.ndim ==1 
    h_prev=  np.atleast_2d ( h_prev )
    x_t = np.atleast_2d ( x_t )

    concat = np.concatenate ( [r_t*h_prev, x_t], axis = 1 )
    h_t = np.tanh (concat@W_h.T + b_h) 
    return h_t[0] if was_1d ==1 else h_t 



