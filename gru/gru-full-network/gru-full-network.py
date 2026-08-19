import torch 
import numpy as np 
import torch.nn as nn 
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)
    def _gru_cell ( self , x_t , h_prev ) : 
        concat = np.concatenate([h_prev, x_t], axis=1)              

        r_t = sigmoid(concat @ self.W_r.T + self.b_r)                
        z_t = sigmoid(concat @ self.W_z.T + self.b_z)                
        concat_reset = np.concatenate([r_t * h_prev, x_t], axis=1)   
        h_tilde = np.tanh(concat_reset @ self.W_h.T + self.b_h)      

        h_t = (1 - z_t) * h_tilde + z_t * h_prev                     
        return h_t


    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass. Returns (y, h_last).
        X shape: (N, T, D)
        y shape: (N, T, output_dim)  -- output tại MỌI timestep
        h_last shape: (N, H)         -- hidden state ở bước cuối
        """
        X = np.asarray(X)
        was_2d = X.ndim == 2
        if was_2d:
            X = X[np.newaxis, :, :]

        N, T, D = X.shape
        h_t = np.zeros((N, self.hidden_dim))

        y_list = []
        for t in range(T):
            x_t = X[:, t, :]
            h_t = self._gru_cell(x_t, h_t)          # (N, H)
            y_t = h_t @ self.W_y.T + self.b_y        # (N, output_dim)
            y_list.append(y_t)

        y = np.stack(y_list, axis=1)                 # (N, T, output_dim)
        h_last = h_t                                  # (N, H)

        if was_2d:
            y = y[0]
            h_last = h_last[0]

        return y, h_last
