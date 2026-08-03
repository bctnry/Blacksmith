import torch
import tiktoken
import numpy as np

Tokenizer = tiktoken.get_encoding('gpt2')

class DatasetManager:
    """
    Dataset manager.
    Memmaps a binary file containing a list of GPT-2 tokenizer token ids.
    All ids needs to be uint16 in little endian.
    """
    def __init__(self, device, p):
        self.dataset_path = p
        self.data = np.memmap(p, dtype=np.uint16, mode='r')
        self.device = device

    def get_batch(self, batch_size, seq_len):
        starts = torch.randint(0, len(self.data) - seq_len - 1, (batch_size,),
                               device=self.device)
        x = torch.stack([torch.tensor(self.data[s:s+seq_len]) for s in starts])
        x = x.to(torch.long).to(self.device)
        y = torch.stack([torch.tensor(self.data[s+1:s+seq_len+1]) for s in starts])
        y = y.to(torch.long).to(self.device)
        return x, y
    
