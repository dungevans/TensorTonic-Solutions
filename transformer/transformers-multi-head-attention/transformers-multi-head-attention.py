import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    batch_size, seq_length, d_model = Q.shape 
    d_k = d_model// num_heads
    Q_projection = np.matmul ( Q, W_q)

    K_projection = np.matmul ( K , W_k )
    V_projection = np.matmul ( V, W_v )
    
    def split_head ( a : np.array, batch_size : int  , seq_length : int, d_model : int , num_head : int  ) -> np.array :
        d_k = d_model //num_head 
        a = a.reshape (batch_size,seq_length,num_head, d_k )
        return a.transpose ( 0,2,1,3)
     
    Q_projection = split_head ( Q_projection, batch_size, seq_length, d_model, num_heads)
    K_projection = split_head ( K_projection, batch_size, seq_length, d_model, num_heads)
    V_projection = split_head ( V_projection, batch_size, seq_length, d_model, num_heads)
    result = np.matmul( Q_projection, K_projection.transpose ( 0,1,3,2))/ np.sqrt ( d_k )
    result = softmax ( result , axis= -1 )
    result = np.matmul ( result, V_projection)
    result = result.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, d_model)
    output = np.matmul(result, W_o)
    
    return output

    