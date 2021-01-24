import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, embedding_matrix, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)

        _, embedding_size = embedding_matrix.shape
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size, device='cpu'):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        vocab_size, embedding_size = embedding_matrix.shape
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = torch.relu(output)
        output = output.unsqueeze(1)

        # input (batch, seq_len, input_size)
        output, hidden = self.gru(output, hidden)
        # output (batch, 1, hidden)
        output = output.squeeze(1) # output (batch, hidden)
        output = self.softmax(self.out(output))
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


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

