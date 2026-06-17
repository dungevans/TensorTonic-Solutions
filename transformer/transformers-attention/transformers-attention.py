import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    # operation = [Q, K , V ]
    # for op in operation : 
    #         if isinstance ( op, torch.Tensor ) : 
    #             op = torch.tensor ( op )

    dk = Q.size (2)
    result = torch.matmul ( Q, K.transpose( 2,1 ))/ math.sqrt ( dk )
    result = F.softmax ( result , dim = -1 )    
    result = torch.matmul ( result, V)
 
    return result 