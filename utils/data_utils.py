import torch


def generate_training_data_reverse_numbers(batch_size, vocab_size, seq_len,device):
    """Generate training data for reverse number prediction task.

    Here the training data isa batch of sequences of integers, where each sequence is a
    sequence of ordered integers from 0 to vocab_size - 1. The target is the reverse of the input

    Args:
        batch_size: Number of samples in the batch.
        vocab_size: Size of the vocabulary (should be at least seq_len).
        seq_len: Length of each sequence.
    """
    assert vocab_size >= seq_len, "vocab_size must be at least seq_len"

    # Randomly sample starting points for sequences
    start_points = torch.randint(0, vocab_size - seq_len + 1, (batch_size,))
    src = torch.stack([torch.arange(start, start + seq_len) for start in start_points], dim=0)
    trgt = torch.flip(src, dims=[1])  # Reverse the sequence
    # Geneerate right shifted trgt
    # Add <eos> and <sos>

    # Ensure that sos_id and eos_id are within the valid range
    sos_id = max(0,  vocab_size - 1)
    eos_id = max(0,  vocab_size - 2)

    # Generate right-shifted target input and target output
    tgt_out = torch.cat([trgt, torch.full((batch_size, 1), eos_id)], dim=1)  # [B, S+1]
    tgt_in = torch.cat([torch.full((batch_size, 1), sos_id), trgt], dim=1)  # [B, S+1]


    return src.to(device), tgt_in.to(device), tgt_out.to(device)
