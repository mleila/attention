import torch
import fasttext

from attention.vocab import Vocabulary
from attention.utils import normalize_string, tokenize_english, tokenize_french
from attention.constants import ENGLISH, FRENCH, SEQ_SIZE


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

            # normalize
            eng_sent = normalize_string(eng_sent)
            fr_sent = normalize_string(fr_sent)

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

        # process sentence
        sentence = normalize_string(sentence)

        # tokenize sentence
        tokens = tokenizer(sentence)

        # build vector
        vector = [vocab.lookup_token(token) for token in tokens]

        # standardize vector length
        if len(vector) < seq_size:
            vector += [vocab.lookup_token(vocab.pad)] * (seq_size - len(vector))
        else:
            vector = vector[:seq_size]

        return torch.tensor(vector)


    def build_embedding_matrix_from_file(self, embedding_fp):
        pass


    def build_embedding_matrix_from_fasttext(self, ft, lang, embedding_size=100):
        # set embedding size
        fasttext.util.reduce_model(ft, embedding_size)

        # select vocab
        vocab = self.english_vocab if lang == ENGLISH else self.french_vocab

        # populate matrix
        matrix = []
        for token in vocab:
            embedding = ft.get_word_vector(token)
            matrix.append(embedding)
        return torch.tensor(matrix)


