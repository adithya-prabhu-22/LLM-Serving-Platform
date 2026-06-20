import torch
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch, pad_token_id: int):
    if len(batch) == 0:
        raise ValueError("Received empty batch.")
    input_ids = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    targets = pad_sequence(targets, batch_first=True, padding_value=-100)
    return input_ids, targets