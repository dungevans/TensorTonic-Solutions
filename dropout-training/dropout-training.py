import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    x_shape = x.shape 


    if rng is None:
        rng = np.random
        

    rand_tensor = rng.uniform(low=0.0, high=1.0, size=x_shape)


    mask = (rand_tensor > p)
    
    dropout_pattern = mask.astype(float) / (1 - p)
    

    out = x * dropout_pattern
    
    
    return out, dropout_pattern