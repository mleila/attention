import os
import re
import unicodedata

import torch

from attention.constants import ENGLISH


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def tokenize_english(sentence):
    return sentence.split(" ")

def tokenize_french(sentence):
    return sentence.split(" ")

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
