import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, embedding_matrix, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        vocab_size, embedding_size = embedding_matrix.shape
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size, device='cpu'):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
