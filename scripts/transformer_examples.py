import torch
import torch.nn as nn
from models.transformer import Transformer, Embedding

from utils.data_utils import generate_training_data_reverse_numbers
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_reverse_transformer(model, data_loader, device, sos_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=sos_idx)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    total_loss = 0.0

    iteration = 0
    print("Training reverse transformer...")
    for src, trgt_in, trgt_out in data_loader:
        print(f"Iteration {iteration} - src: {src.shape}, trgt_in: {trgt_in.shape}, trgt_out: {trgt_out.shape}")
        src, trgt_in, trgt_out = src.to(device), trgt_in.to(device), trgt_out.to(device)

        src = embedding(src)
        trgt_in = embedding(trgt_in)

        optimizer.zero_grad()
        output = model(src, trgt_in)

        # Compute loss
        loss = criterion(output.view(-1, output.size(-1)), trgt_out.view(-1))
        loss.backward(retain_graph=True)  # Retain graph for next iteration
        optimizer.step()

        total_loss += loss.item()
        iteration += 1

    return total_loss / len(data_loader)


def test_trained_revserse_transform(model,embedding, vocab_size, seq_len, device):
    """Test the trained reverse transformer model on a simple task."""
    batch_size = 25

    # Generate a single test sequence
    src, tgt_in, tgt_out = generate_training_data_reverse_numbers(batch_size, vocab_size, seq_len,
                                                                  device=device)

    # Embed src and tgt_in
    src = embedding(src)
    tgt_in = embedding(tgt_in)

    model.eval()
    with torch.no_grad():
        output = model(src, tgt_in)

    # Check if the output matches the expected target output
    predicted = torch.argmax(output, dim=-1)
    for i in range(batch_size):
        
        # print("Source:", src[i])
        # print("Target Input:", tgt_in[i])
        print("Target Output:", tgt_out[i])
        print("Predicted Output:", predicted[i])
        print("\n")

if __name__ == '__main__':
    vocab_size = 20
    sos_idx = vocab_size - 1  # Assuming the last index is used for <sos>
    eos_idx = vocab_size - 2  # Assuming the last index is used for <eos>
    d_embed = 8
    seq_len = 8
    batch_size = 512
    n_batches = 1
    # Embed src and tgt_in
    embedding = Embedding(vocab_size, batch_size, d_embed)

    # data_loader = [(src, tgt_in, tgt_out)]  # Simple data loader for demonstration

    model = Transformer(vocab_size, d_embed, d_embed, 0.1)
    num_epochs = 2000

    for epoch in range(num_epochs):
        data_loader = [(generate_training_data_reverse_numbers(batch_size, vocab_size, seq_len,
                                                                  device=device,
                                                                  )) for _ in range(n_batches)]
        loss = train_reverse_transformer(model, data_loader, device, sos_idx)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}\n")



    # Test the trained model
    test_trained_revserse_transform(model, embedding, vocab_size, seq_len, device)
