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


class TranslationModel(nn.Module):

    def __init__(self, encoder, decoder, output_vocab_size):
        super(TranslationModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.output_vocab_size = output_vocab_size

    def train_forward(self, encoder_input, decoder_input, device='cpu'):
        """
        Model forward

        encoder_input: (batch_size, seq_len)
        decoder_input: (batch_size, seq_len)
        """
        # encoder setup
        batch_size, seq_len = encoder_input.shape
        encoder_hidden = self.encoder.init_hidden(batch_size)

        # encoder forward pass
        encoder_output, encoder_hidden = self.encoder(encoder_input, encoder_hidden)
        encoder_output = encoder_output.to(device)
        encoder_hidden = encoder_hidden.to(device)

        # decoder setup
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros(seq_len, batch_size, self.output_vocab_size)

        # decoder forward pass
        for index in range(seq_len-1):
            # select input tokens (batch_size, 1)
            current_tokens = decoder_input[:, index]
            decoder_output, decoder_hidden = self.decoder(current_tokens, decoder_hidden)
            decoder_outputs[index] = decoder_output

        return decoder_outputs.view(batch_size, seq_len, self.output_vocab_size)

    def eval_forward(self, encoder_input, sos_token, device):

        with torch.no_grad():

            # encoder setup
            batch_size, seq_len = encoder_input.shape
            encoder_hidden = self.encoder.init_hidden(batch_size, device=device)

            # encoder forward pass
            encoder_output, encoder_hidden = self.encoder(encoder_input, encoder_hidden)
            encoder_output = encoder_output.to(device)

            # decoder setup
            decoder_hidden = encoder_hidden
            decoder_outputs = torch.zeros(batch_size, seq_len, self.output_vocab_size)
            next_tokens = torch.tensor(sos_token).expand(batch_size)

            # decoder forward pass
            for token in range(seq_len-1):
                decoder_output, decoder_hidden = self.decoder(next_tokens, decoder_hidden)
                decoder_outputs[:, token] = decoder_output
                next_tokens = torch.argmax(decoder_output, dim=1)

            return decoder_outputs

    def forward(self, encoder_input, decoder_input=None, sos_token=None, device='cpu'):
        if self.training:
            return self.train_forward(encoder_input, decoder_input, device)
        return self.eval_forward(encoder_input, sos_token, device)

