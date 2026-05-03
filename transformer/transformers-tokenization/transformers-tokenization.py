import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
      
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then sorted unique words.
        """
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

       
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx  
            self.id_to_word[idx] = token  

        self.vocab_size = len(special_tokens)
        
        
        unique_words = set()
        for text in texts: 
            words = text.lower().split() 
            for word in words:
                unique_words.add(word) 
                

        sorted_words = sorted(list(unique_words))
        

        for word in sorted_words:
            current_id = self.vocab_size  
            self.word_to_id[word] = current_id 
            self.id_to_word[current_id] = word 
            self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        token_ids = []
        
 
        
        words = text.lower().split() 
        for word in words: 
            if word not in self.word_to_id: 
                
                token_ids.append(self.word_to_id[self.unk_token])
            else:
             
                token_ids.append(self.word_to_id[word])
                


        
        return token_ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        (UNK for unknown IDs)
        """
        words = []
        
        for i in ids: 

            word = self.id_to_word.get(i, self.unk_token)
            
          
            if word in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            
            words.append(word) 
            
        return " ".join(words)
