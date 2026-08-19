import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def gru_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
             b_r: np.ndarray, b_z: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Complete GRU cell forward pass.
    """
    h_prev = np.asarray(h_prev)
    x_t = np.asarray(x_t)
    W_r = np.asarray ( W_r )
    W_z = np.asarray ( W_z)
    W_h= np.asarray ( W_h )
    

    
    was_1d = h_prev.ndim == 1

    h_prev = np.atleast_2d(h_prev)
    x_t = np.atleast_2d(x_t)
    concat = np.concatenate([h_prev, x_t], axis=1)          # (N, H+D)

    r_t = sigmoid(concat @ W_r.T + b_r)                      # (N, H)

    z_t = sigmoid(concat @ W_z.T + b_z)                      # (N, H)

    concat_reset = np.concatenate([r_t * h_prev, x_t], axis=1)  # (N, H+D)
    h_tilde = np.tanh(concat_reset @ W_h.T + b_h)                # (N, H)

    
    h_t = (1 - z_t) * h_tilde + z_t * h_prev                 # (N, H)

    return h_t[0] if was_1d else h_t