import torch

class KVCache:
    def __init__(self, batch_size: int, num_heads: int, head_dim: int, max_seq_len: int, device, dtype):
        self.max_seq_len = max_seq_len
        self.keys = torch.empty((batch_size, num_heads, max_seq_len, head_dim), device=device, dtype=dtype)
        self.values = torch.empty((batch_size, num_heads, max_seq_len, head_dim), device=device, dtype=dtype)
        self.length = 0

    def append(self, new_keys: torch.Tensor, new_values: torch.Tensor):
        seq_len = new_keys.size(2)
        if seq_len > self.max_seq_len:
            raise ValueError("Incoming sequence exceeds cache capacity.")
        if self.length + seq_len <= self.max_seq_len:
            start = self.length
            end = start + seq_len
            self.keys[:, :, start:end, :] = new_keys
            self.values[:, :, start:end, :] = new_values
            self.length = end
            return
        overflow = self.length + seq_len - self.max_seq_len
        self.keys[:, :, :-overflow, :] = self.keys[:, :, overflow:, :].clone()
        self.values[:, :, :-overflow, :] = self.values[:, :, overflow:, :].clone()
        self.keys[:, :, self.max_seq_len - seq_len:, :] = new_keys
        self.values[:, :, self.max_seq_len - seq_len:, :] = new_values
        self.length = self.max_seq_len

    def get_keys(self):
        return self.keys[:, :, :self.length, :]

    def get_values(self):
        return self.values[:, :, :self.length, :]

    def get_kv(self):
        return self.get_keys(), self.get_values()

    def reset(self):
        self.length = 0

    def __len__(self):
        return self.length