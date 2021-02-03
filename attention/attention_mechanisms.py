"""This module implements different attention mechanisms."""
import torch


def dotproduct_attention(encoder_state_vectors, query_vector):
    """
    encoder_state_vectors: 3dim tensor from bi-GRU in encoder
        expected_dim = (batch_size, seq_size, encoder_hidden_size)
    query_vector: hidden_state from current step in decoder
        expected_dim = (batch_size, decoder_hidden_size)
    """
    # query vector (batch_size, decoder_hidden_size) -> (batch_size, hidden_size, 1)
    query_vector = query_vector.unsqueeze(2)

    # (batch_size, seq_size, encoder_hidden_size) x (batch_size, decoder_hidden_size)
    # -> (batch_size, seq_size, 1)
    vector_scores = torch.matmul(encoder_state_vectors, query_vector)

    # vector scores (batch_size, seq_size, 1) -> (batch_size, seq_size)
    vector_scores = vector_scores.squeeze(-1)

    vector_probabilities = torch.softmax(vector_scores, dim=-1)

    # (batch_size, seq_size (-2), encoder_hidden_size (-1)) ->
    # (batch_size, encoder_hidden_size, seq_size)
    encoder_state_vectors = encoder_state_vectors.transpose(-2, -1)

    # (batch_size, seq_size) -> (batch_size, seq_size, 1)
    squeezed_vector_prob = vector_probabilities.unsqueeze(2)

    # context_vector (batch_size, encoder_hidden_size, 1)
    context_vectors = torch.matmul(
        encoder_state_vectors,
        squeezed_vector_prob
    )
    # (batch_size, encoder_hidden_size, 1) -> (batch_size, encoder_hidden_size)
    context_vectors = context_vectors.squeeze(-1)

    return context_vectors, vector_probabilities
