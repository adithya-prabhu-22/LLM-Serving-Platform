import torch


class RingKVCache:

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        device=None,
        dtype=None,
    ):

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.capacity = max_seq_len

        self.device = device
        self.dtype = dtype

        self.keys = torch.empty(
            (
                batch_size,
                num_heads,
                max_seq_len,
                head_dim,
            ),
            device=device,
            dtype=dtype,
        )

        self.values = torch.empty(
            (
                batch_size,
                num_heads,
                max_seq_len,
                head_dim,
            ),
            device=device,
            dtype=dtype,
        )

        self.write_pos = 0

        self.length = 0

        self.total_tokens_seen = 0

    def append(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ):

        seq_len = new_keys.size(2)

        for i in range(seq_len):

            self.keys[
                :,
                :,
                self.write_pos,
                :
            ] = new_keys[
                :,
                :,
                i,
                :
            ]

            self.values[
                :,
                :,
                self.write_pos,
                :
            ] = new_values[
                :,
                :,
                i,
                :
            ]

            self.write_pos = (
                self.write_pos + 1
            ) % self.capacity

            self.length = min(
                self.length + 1,
                self.capacity,
            )

            self.total_tokens_seen += 1

    def get_keys(self):

        if self.length == 0:

            return torch.empty(
                (
                    self.batch_size,
                    self.num_heads,
                    0,
                    self.head_dim,
                ),
                device=self.device,
                dtype=self.dtype,
            )

        if self.length < self.capacity:

            return self.keys[
                :,
                :,
                :self.length,
                :
            ]

        return torch.cat(
            (
                self.keys[
                    :,
                    :,
                    self.write_pos:,
                    :
                ],
                self.keys[
                    :,
                    :,
                    :self.write_pos,
                    :
                ],
            ),
            dim=2,
        )

    def get_values(self):

        if self.length == 0:

            return torch.empty(
                (
                    self.batch_size,
                    self.num_heads,
                    0,
                    self.head_dim,
                ),
                device=self.device,
                dtype=self.dtype,
            )

        if self.length < self.capacity:

            return self.values[
                :,
                :,
                :self.length,
                :
            ]

        return torch.cat(
            (
                self.values[
                    :,
                    :,
                    self.write_pos:,
                    :
                ],
                self.values[
                    :,
                    :,
                    :self.write_pos,
                    :
                ],
            ),
            dim=2,
        )

    def get_kv(self):

        return (
            self.get_keys(),
            self.get_values(),
        )

    def get_total_tokens_seen(self):

        return self.total_tokens_seen

    def reset(self):

        self.write_pos = 0

        self.length = 0

        self.total_tokens_seen = 0

    def __len__(self):

        return self.length