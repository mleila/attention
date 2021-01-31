import torch


def dotproduct_attention(encoder_state_vectors, query_vector):
    """
    encoder_state_vectors: 3dim tensor from bi-GRU in encoder
    query_vector: hidden_state from current step in decoder
    """
    vector_scores = torch.matmul(encoder_state_vectors, query_vector.unsqueeze(2)).squeeze()
    vector_probabilities = torch.softmax(vector_scores, dim=-1)
    context_vectors = torch.matmul(
        encoder_state_vectors.transpose(-2, -1),
        vector_probabilities.unsqueeze(2)
    ).squeeze()
    return context_vectors, vector_probabilities
