import torch
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size=None, embedding_matrix=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # embedding layer
        if embedding_matrix is None:
            self.embedding = nn.Embedding(input_size, embedding_size)
        else:
            embedding_size = embedding_matrix.shape[-1]
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)

        # rnn layer
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input, hidden):

        embedded = self.embedding(input)
        output = embedded.unsqueeze(0)
        # GRU input (seq_len, batch, input_size)
        # GRU hidden (num_layers * num_directions, batch, hidden_size)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size, device):
        # (num_layers * num_directions, batch_size, hidden_size)
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_size=None, embedding_matrix=None):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # embedding layer
        if embedding_matrix is None:
            self.embedding = nn.Embedding(output_size, embedding_size)
        else:
            embedding_size = embedding_matrix.shape[-1]
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)

        # rnn layer
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
