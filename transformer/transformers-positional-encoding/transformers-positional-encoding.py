import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    position = np.arange ( 0, seq_length )
    cols = np.arange ( 0, d_model )
    W_PE = np.zeros((seq_length, d_model ))
    
    cols_prepare = cols//2 
    common = np.exp(np.log(10000)*2*cols_prepare/d_model )
    for col in cols : 
        if col % 2 == 0 : 
            W_PE [:, col] = np.sin ( position/common[col])
        else :
            W_PE [:, col] = np.cos ( position/common[col])

    return W_PE 
#test 
