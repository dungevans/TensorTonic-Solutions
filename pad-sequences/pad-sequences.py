import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):

    if max_len is None:
        max_len = max(len(s) for s in seqs)
    
    res = np.full((len(seqs), max_len), pad_value)

    for i, seq in enumerate(seqs):

        length = min(len(seq), max_len)
        res[i, :length] = seq[:length]
        
    return res

