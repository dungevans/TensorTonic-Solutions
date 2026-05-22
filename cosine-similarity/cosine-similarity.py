import numpy as np 
def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.array ( a )
    b = np.array (b)
    a_norm =  np.sqrt ( np.sum(a**2))
    b_norm = np.sqrt ( np.sum ( b**2))
    result = ((a@b)/(a_norm*b_norm))
    if  a_norm == 0 or b_norm==0 : return 0  
    return result