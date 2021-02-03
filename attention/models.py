"""This Module contain pytorch models"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from attention.attention_mechanisms import dotproduct_attention
from attention.constants import SEQ_SIZE


class EncoderRNN(nn.Module):
    """
    Regular RNN Encoder with a GRU RNN.
    """
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
    """
    Regular decoder (no attention)
    """
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


class BiDirectionalEncoder(nn.Module):
    """
    Encoder with a bidirectional GRU
    """
    def __init__(
        self,
         num_embeddings: int,
         embedding_size: int,
         rnn_hidden_size: int):
        """
        Args:
            num_embeddings: size of source vocab
            embedding_size: size of embedding vectors
            rnn_hidden_size: size of rnn hidden state vectors
        """
        super(BiDirectionalEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=0)
        self.bi_rnn = nn.GRU(embedding_size, rnn_hidden_size, bidirectional=True, batch_first=True)

    def forward(
        self, x_source, x_lengths):
        """
        The forward pass of the encoder

        Args:
            x_source: the input data tensor
                x_source shape should be (batch_size, seq_size)
            x_lengths: vector of lengths for each item in the batch
        """
        x_embedded = self.embedding(x_source)

        # create PackedSequence
        # x_packed.data.shape = (number of items, embedding_size)
        x_lengths = x_lengths.detach().cpu().numpy()
        x_packed = pack_padded_sequence(x_embedded, x_lengths, batch_first=True)

        # x_bi_rnn_h.shape = (num_rnn, batch_size, feature_size)
        x_bi_rnn_out, x_bi_rnn_h = self.bi_rnn(x_packed)
        # permute to (batch_size, num_rnn, feature_size)
        x_bi_rnn_h = x_bi_rnn_h.permute(1, 0, 2)

        # flatten features; reshape to (batch_size, num_rnn * feature_size)
        num_rnn = x_bi_rnn_h.size(0)
        x_bi_rnn_h = x_bi_rnn_h.contiguous().view(num_rnn, -1)

        x_unpacked, _ = pad_packed_sequence(x_bi_rnn_out, batch_first=True)
        return x_unpacked, x_bi_rnn_h


class NMTDecoder(nn.Module):
    """
    Decoder with attention.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_size: int,
        rnn_hidden_size: int,
        sos_index: int

    ):
        super(NMTDecoder, self).__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self._num_embeddings = num_embeddings

        self.target_embedding = nn.Embedding(
            num_embeddings,
            embedding_size,
            padding_idx=0
        )
        self.gru_cell = nn.GRUCell(
            embedding_size + rnn_hidden_size,
            rnn_hidden_size
        )
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size*2, num_embeddings)
        self.sos_index = sos_index

    def _init_indices(self, batch_size):
        """return the begin of sequence index vector"""
        return torch.ones(batch_size, dtype=torch.int64) * self.sos_index

    def _init_context_vectors(self, batch_size):
        """returns a zeros vector for initializing context vectors"""
        return torch.zeros(batch_size, self._rnn_hidden_size)

    def _train_forward(
        self,
        encoder_state: torch.Tensor,
        initial_hidden_state: torch.Tensor,
        target_sequence: torch.Tensor
    ):
        """
        The forward pass of the model

        Args:
            encoder_state: output of the NMT Encoder
            initial_hidden_state: final hidden state of the NMT encoder
            target_sequence: target text data tensor (batch_size, seq_size)
        Returns:
            output_vectors: prediction vectors at each output step
        """
        # we want to iterate over the sequence, so we will flip it's dimensions
        # from (batch_size, seq_size) -> (seq_size, batch_size)
        target_sequence = target_sequence.permute(1, 0)

        # use the provided encoder hidden state as the initial hidden state
        h_t = self.hidden_map(initial_hidden_state)

        batch_size = encoder_state.size(0)
        # initialize context vectors
        context_vectors = self._init_context_vectors(batch_size)
        # initialize first y_t words as start of sentence indices
        y_t_index = self._init_indices(batch_size)

        # attach to device
        h_t = h_t.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)

        output_vectors = []
        # all cached tensors are moved from the GPU and stored for analysis
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()

        output_sequence_size = target_sequence.size(0)
        for i in range(output_sequence_size):

            # step 1: embed word and concat with previous context
            y_input_vector = self.target_embedding(target_sequence[i])
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)

            # step 2: make a GRU step, getting a new hidden_state
            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().data.numpy())

            # step 3: use current hidden state to attend to encoder states
            context_vectors, p_attn = dotproduct_attention(
                encoder_state_vectors=encoder_state,
                query_vector=h_t
            )

            # cache the attention probabilities for visualization
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())

            # step 4: use current hidden and context vectors
            prediction_vector = torch.cat([context_vectors, h_t], dim=1)
            score_for_y_t_index = self.classifier(prediction_vector)

            # collect prediction socres
            output_vectors.append(score_for_y_t_index)

        return torch.stack(output_vectors)

    def _eval_forward(self,
                      encoder_state: torch.Tensor,
                      initial_hidden_state: torch.Tensor,
                      target_sequence: torch.Tensor
                     ):
        batch_size = encoder_state.size(0)
        seq_size = target_sequence.size(1)

        output_sequence = torch.zeros(seq_size, batch_size, self._num_embeddings, dtype=torch.int64)
        output_sequence[0, :, :] = self.sos_index


        # use the provided encoder hidden state as the initial hidden state
        h_t = self.hidden_map(initial_hidden_state)

        # initialize context vectors
        context_vectors = self._init_context_vectors(batch_size)
        # initialize first y_t words as start of sentence indices
        y_t_index = self._init_indices(batch_size)

        # attach to device
        h_t = h_t.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)

        # all cached tensors are moved from the GPU and stored for analysis
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()

        outputs = []
        curr_out =  torch.ones(batch_size, dtype=torch.int64) * self.sos_index
        for i in range(seq_size):

            # step 1: embed word and concat with previous context
            y_input_vector = self.target_embedding(curr_out)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)

            # step 2: make a GRU step, getting a new hidden_state
            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().data.numpy())

            # step 3: use current hidden state to attend to encoder states
            context_vectors, p_attn = dotproduct_attention(
                encoder_state_vectors=encoder_state,
                query_vector=h_t
            )

            # cache the attention probabilities for visualization
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())

            # step 4: use current hidden and context vectors
            prediction_vector = torch.cat([context_vectors, h_t], dim=1)
            score_for_y_t_index = self.classifier(prediction_vector)
            curr_out = torch.argmax(score_for_y_t_index, dim=-1)
            outputs.append(score_for_y_t_index)

        return torch.stack(outputs)

    def forward(self, encoder_state, initial_hidden_state, target_sequence):
        if self.training:
            return self._train_forward(encoder_state, initial_hidden_state, target_sequence)
        return self._eval_forward(encoder_state, initial_hidden_state, target_sequence)


class AttentionModel(nn.Module):
    """
    Full Seq2Seq model with attention.
    """

    def __init__(
        self,
        source_vocab_size: int,
        source_embedding_size: int,
        target_vocab_size: int,
        target_embedding_size: int,
        encoding_size: int,
        target_sos_index: int
        ):
        """
        Args:
            source_vocab_size: number of unique words in source language
            source_embedding_size: size of the source embedding vectors
            target_vocab_size: number of unique words in target language
            target_embedding_size: size of the target embedding vectors
            encoding_size: size of the encoder RNN (hidden state size)
            target_sos_index: index of start of sentence token
        """
        super().__init__()

        self.encoder = BiDirectionalEncoder(
            num_embeddings=source_vocab_size,
            embedding_size=source_embedding_size,
            rnn_hidden_size=encoding_size
        )

        # becaus we concat decoder hidden state with context vector
        decoding_size = encoding_size * 2
        self.decoder = NMTDecoder(
            num_embeddings=target_vocab_size,
            embedding_size=target_embedding_size,
            rnn_hidden_size=decoding_size,
            sos_index=target_sos_index
        )

    def forward(
        self,
        x_source: torch.Tensor,
        x_source_lengths: torch.Tensor,
        target_sequence: torch.Tensor
    ):
        """
        The forward pass of the model

        Args:
            x_source: the source text data tensor
                x_source.shape should be (batch_size, vectorizer.max_source_length)
            x_source_lengths: the length of sequences in x_source
            target_sequence: the target text data tensor
        Returns:
            decoded_states: prediction vectors at each output step
        """
        encoder_state, final_hidden_state = self.encoder(x_source, x_source_lengths)
        decoded_states = self.decoder(
            encoder_state=encoder_state,
            initial_hidden_state=final_hidden_state,
            target_sequence=target_sequence
        )
        return decoded_states
