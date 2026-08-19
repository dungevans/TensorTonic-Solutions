import numpy as np 
def _sigmoid ( x ) : 
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))



def update_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_z: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    """
    Compute update gate: z_t = sigmoid(W_z @ [h, x] + b_z)
    """
    
    h_prev  = np.asarray ( h_prev)
    x_t = np.asarray ( x_t)
    W_z = np.asarray ( W_z )
    b_z = np.asarray ( b_z )

    was_1d = h_prev.ndim == 1 
    
    h_prev = np.atleast_2d(h_prev)
    x_t = np.atleast_2d(x_t)
   
    concat = np.concatenate ( [h_prev , x_t ], axis = 1)
    z_t = concat @ W_z.T + b_z
    result = _sigmoid(z_t)
    return result[0] if was_1d else result