import torch
import numpy as np

from attention.vocab import Vocabulary
from attention.text_preprocessing import preprocess_text, tokenize_english, tokenize_french
from attention.constants import ENGLISH, FRENCH, SEQ_SIZE


def source_tokenizer(sentence):
    return sentence.split(" ")

def target_tokenizer(sentence):
    return sentence.split(" ")


class Vectorizer:

    def __init__(self, english_vocab, french_vocab):
        self.english_vocab = english_vocab
        self.french_vocab = french_vocab

    @classmethod
    def from_df(cls, df):

        english_vocab = Vocabulary()
        french_vocab = Vocabulary()

        for _, row in df.iterrows():
            eng_sent, fr_sent = row[ENGLISH], row[FRENCH]

            # tokens
            for en_token in eng_sent.split(" "):
                english_vocab.add_token(en_token)

            for fr_token in fr_sent.split(" "):
                french_vocab.add_token(fr_token)

        return cls(english_vocab, french_vocab)


    def vectorize_sentence(self, sentence, language, seq_size=SEQ_SIZE):

        # select vocab
        vocab = self.english_vocab if language == ENGLISH else self.french_vocab

        # select tokenizer
        tokenizer = tokenize_english if language == FRENCH else tokenize_french

        # tokenize sentence
        tokens = tokenizer(sentence)

        # build vector
        vector = [vocab.lookup_token(vocab.sos)]
        vector += [vocab.lookup_token(token) for token in tokens]
        vector += [vocab.lookup_token(vocab.eos)]

        # standardize vector length
        if len(vector) < seq_size:
            vector += [vocab.lookup_token(vocab.pad)] * (seq_size - len(vector))
        else:
            vector = vector[:seq_size-1] + [vocab.lookup_token(vocab.eos)]

        return torch.tensor(vector)


    def build_embedding_matrix_from_file(self, embedding_fp):
        pass


    def build_embedding_matrix_from_spacy(self, model, lang):
        # select vocab
        vocab = self.english_vocab if lang == ENGLISH else self.french_vocab

        # populate matrix
        matrix = []
        for token in vocab:
            spacy_token = model(token)
            embedding = spacy_token.vector
            if embedding.size == 0:
                continue
            matrix.append(embedding)
        return torch.tensor(matrix)


    def build_embedding_matrix_from_fasttext(self, model, lang):
        # select vocab
        vocab = self.english_vocab if lang == ENGLISH else self.french_vocab

        # populate matrix
        matrix = []
        for token in vocab:
            embedding = model.get_word_vector(token)
            matrix.append(embedding)
        return torch.tensor(matrix)



class NMTVectorizer:
    """Coordinates Vocabularies and puts them to use."""

    def __init__(self, source_vocab, target_vocab, max_source_length, max_target_length):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    @classmethod
    def from_df(cls, df):
        source_vocab = Vocabulary()
        target_vocab = Vocabulary()
        max_source_length = 0
        max_target_length = 0

        for _, row in df.iterrows():
            source_tokens = tokenize_english(row[ENGLISH])
            max_source_length = max(max_source_length, len(source_tokens))
            for token in source_tokens:
                source_vocab.add_token(token)

            target_tokens = tokenize_french(row[FRENCH])
            max_target_length = max(max_target_length, len(target_tokens))
            for token in target_tokens:
                target_vocab.add_token(token)
        return cls(source_vocab, target_vocab, max_source_length, max_target_length)

    def _vectorize(self, indices, vector_length=-1, mask_index=0):
        """Vectorize the provided indices."""
        if vector_length < 0:
            vector_length = len(indices)
        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index
        return vector

    def _get_source_indices(self, text):
        normalized = preprocess_text(text)
        tokens = source_tokenizer(normalized)

        indices = [self.source_vocab.sos_index]
        indices.extend(self.source_vocab.lookup_token(token) for token in tokens)
        indices.append(self.source_vocab.eos_index)
        return indices

    def _get_target_indices(self, text):
        normalized = preprocess_text(text)
        tokens = target_tokenizer(normalized)
        core = [self.target_vocab.lookup_token(token) for token in tokens]
        x_indices = [self.target_vocab.sos_index] + core
        y_indices = core + [self.target_vocab.eos_index]
        return x_indices, y_indices

    def vectorize(self, source_text, target_text, use_dataset_max_len=True):
        source_vector_length = -1
        target_vector_length = -1
        if use_dataset_max_len:
            source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_target_length + 1

        source_indices = self._get_source_indices(source_text)
        source_vector  = self._vectorize(
            source_indices, source_vector_length, mask_index=self.source_vocab.pad_index)

        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(
            target_x_indices, target_vector_length, mask_index=self.target_vocab.pad_index)
        target_y_vector = self._vectorize(
            target_y_indices, target_vector_length, mask_index=self.target_vocab.pad_index)
        return {
            "source_vector": source_vector,
            "target_x_vector": target_x_vector,
            "target_y_vector": target_y_vector,
            "source_length": len(source_indices)
        }
