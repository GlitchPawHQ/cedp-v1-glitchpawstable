import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text, add_special_tokens=True)

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, index):
        start = index * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]

        if len(chunk) < self.seq_len + 1:
            chunk += [0] * (self.seq_len + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
