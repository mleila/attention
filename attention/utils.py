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

def translate(sentence, model, vectorizer):
    french_vocab = vectorizer.french_vocab
    encoder_input = vectorizer.vectorize_sentence(sentence, language=ENGLISH).unsqueeze(0)
    model.eval()
    sos_token = french_vocab.lookup_token(french_vocab.sos)
    prediction = model(encoder_input, sos_token=sos_token).squeeze(0)
    word_indices = torch.argmax(prediction, dim=1)
    words = [vectorizer.french_vocab.lookup_index(idx) for idx in word_indices]
    return ' '.join(words)