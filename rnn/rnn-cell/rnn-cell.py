import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    W_xh_tranpose = W_xh.transpose()
    W_hh_tranpose = W_hh.transpose()
    ht = np.tanh ( x_t@W_xh_tranpose + h_prev@W_hh_tranpose+b_h )
    return ht 