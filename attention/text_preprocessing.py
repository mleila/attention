"""This moudle contains text preprocessing logic."""
import re
import unicodedata


def unicode_to_ascii(text):
    """Convert all unicode text to ascii."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def preprocess_text(text):
    """Entry point to preprocessing logic"""
    # lowercase all
    text = text.lower()

    # remove extra whitespaces
    text = text.strip()

    # convert from unicode to ascii
    text = unicode_to_ascii(text)

    # regex rules to handle punctuation
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)

    return text


def tokenize_english(sentence):
    """English language tokenizer."""
    return sentence.split(" ")


def tokenize_french(sentence):
    """French language tokenizer """
    return sentence.split(" ")
