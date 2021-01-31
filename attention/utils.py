"""General utilities module."""
import os

import torch

from attention.constants import ENGLISH


def handle_dirs(dirpath):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)



def translate_simple_rnn(sentence, encoder, decoder, vectorizer, device):
    batch_size = 1
    english_vector = vectorizer.vectorize_sentence(sentence, language=ENGLISH).to(device)
    french_vocab = vectorizer.french_vocab
    seq_size = len(english_vector)

    sos_token = vectorizer.french_vocab.sos
    eos_token = vectorizer.french_vocab.eos

    encoder_hidden = encoder.initHidden(batch_size, device)
    for idx in range(seq_size):
        token = english_vector[idx].unsqueeze(0)
        _, encoder_hidden = encoder(token, encoder_hidden)

    decoder_outputs = torch.zeros(seq_size-1, decoder.output_size)

    hidden = encoder_hidden
    sos_token_index = int(french_vocab.lookup_token(sos_token))
    eos_token_index = int(french_vocab.lookup_token(eos_token))
    output = torch.tensor([sos_token_index], dtype=torch.int64).view(1, 1)

    indices = []
    for idx in range(seq_size-1):
        token = english_vector[idx].view(batch_size, 1)
        token_word = french_vocab.lookup_index(token[0].item())
        if token_word == eos_token:
            break

        output, hidden =  decoder(output, hidden)
        output = torch.argmax(output, dim=-1).view(1, 1)
        if output.squeeze(0).item() == eos_token_index:
            break
        indices.append(output.squeeze(0).item())
        decoder_outputs[idx] = output.squeeze(0)

    words = [french_vocab.lookup_index(idx) for idx in indices]
    return ' '.join(words)


def translate_attention_rnn(sentence, encoder, decoder, vectorizer, device):
    batch_size = 1
    english_vector = vectorizer.vectorize_sentence(sentence, language=ENGLISH).to(device)
    french_vocab = vectorizer.french_vocab
    seq_size = len(english_vector)

    sos_token = vectorizer.french_vocab.sos
    eos_token = vectorizer.french_vocab.eos

    encoder_hidden = encoder.initHidden(batch_size, device)
    encoder_outputs = torch.zeros(seq_size-1, batch_size, encoder.hidden_size)
    for idx in range(seq_size-1):
        token = english_vector[idx].unsqueeze(0)
        _, encoder_hidden = encoder(token, encoder_hidden)
        encoder_outputs[idx] = encoder_hidden

    encoder_outputs = encoder_outputs.view(batch_size, seq_size-1, encoder.hidden_size)

    decoder_outputs = torch.zeros(seq_size-1, decoder.output_size)

    hidden = encoder_hidden
    sos_token_index = int(french_vocab.lookup_token(sos_token))
    eos_token_index = int(french_vocab.lookup_token(eos_token))
    output = torch.tensor([sos_token_index], dtype=torch.int64).view(1, 1)

    indices = []
    for idx in range(seq_size-1):
        token = english_vector[idx].view(batch_size, 1)
        token_word = french_vocab.lookup_index(token[0].item())
        if token_word == eos_token:
            break

        output, hidden =  decoder(output, hidden, encoder_outputs, step=idx)
        output = torch.argmax(output, dim=-1).view(1, 1)
        if output.squeeze(0).item() == eos_token_index:
            break
        indices.append(output.squeeze(0).item())
        decoder_outputs[idx] = output.squeeze(0)

    words = [french_vocab.lookup_index(idx) for idx in indices]
    return ' '.join(words)
