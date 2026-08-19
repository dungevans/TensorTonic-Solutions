import numpy as np

def _sigmoid(x):

    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

import numpy as np

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray, W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    h_prev = np.asarray(h_prev)
    x_t = np.asarray(x_t)

    was_1d = h_prev.ndim == 1

    h_prev = np.atleast_2d(h_prev)
    x_t = np.atleast_2d(x_t)

    concat = np.concatenate([h_prev, x_t], axis=1)   # (N, H+D)
    z = concat @ W_r.T + b_r                          # (N, H)
    r_t = _sigmoid ( z )

    return r_t[0] if was_1d else r_t