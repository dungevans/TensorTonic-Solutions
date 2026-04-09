import numpy as np 


def positional_encoding(seq_len: int, d_model: int, base_line: int = 10000) -> np.ndarray: 
    
    result = np.zeros((seq_len, d_model))
    
   
    positions = np.arange(0, seq_len)
    
    for col in range(d_model): 
        
        i = col // 2
        denominator = np.exp(2 * i * np.log(base_line) / d_model)
        
        
        if col % 2 == 0:
            result[:, col] = np.sin(positions / denominator)
        else:
            result[:, col] = np.cos(positions / denominator)
            
    return result