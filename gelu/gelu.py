import math
import numpy as np

def gelu(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.asarray (x ) 
    y = x.flatten() 
    y = y / math.sqrt ( 2 )
    itermediate_x  = np.asarray ([math.erf ( y_i )for y_i in y ]) 
    itermediate_x = itermediate_x.reshape(x.shape)
    return ( (x/2)*(1+(itermediate_x) )) 

 

