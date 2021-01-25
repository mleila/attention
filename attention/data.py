from numpy.lib.function_base import vectorize
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from attention.vectorizer import Vectorizer
from attention.constants import (
    ENGLISH, FRENCH, SEQ_SIZE,
    SPLIT,
    TRAIN,
    VALID,
    TEST,
    ENCODER_INPUT,
    DECODER_INPUT,
    ENGLISH,
    FRENCH
    )


def load_sentences_dataframe(fp):
    df = pd.read_csv(fp, sep='\t', header=None)
    df.columns = ['english', 'french']
    return df

def assign_rows_to_split(
    df: pd.DataFrame,
    train_ratio: float=0.7,
    valid_ratio: float=0.15,
    test_ratio: float=0.15
    ):
    """
    Assign each row to either a training, validation, or testing datasets
    Args:
     - df: pandas dataframe with two columns, news headline and label
     - train_ratio: ratio of training data
     - valid_ratio: ratio of validation data
     - test_ratio: ratio of testing data
    returns:
     - dataframe with an added column (dataset) with either train, test, or valid
    """
    assert train_ratio + valid_ratio + test_ratio == 1, 'splitting ratios must add to one'

    train_rows, non_train_rows = train_test_split(
        df,
        train_size=train_ratio,
        shuffle=True,
        )

    valid_rows, test_rows = train_test_split(
        non_train_rows,
        train_size=valid_ratio/(valid_ratio+test_ratio),
        )

    train_rows[SPLIT] = TRAIN
    valid_rows[SPLIT] = VALID
    test_rows[SPLIT] = TEST
    return pd.concat([train_rows, valid_rows, test_rows], axis=0)


class TranslationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        vectorizer: Vectorizer,
        max_seq_length: int=SEQ_SIZE
        ):
        self.df = df.reset_index(drop=True)
        self.vectorizer = vectorizer
        self.max_seq_length = max_seq_length
        self.set_split()

    @classmethod
    def from_dataframe(cls, df):
        vectorizer = Vectorizer.from_df(df)
        return cls(df, vectorizer)

    def set_split(self, split: str=TRAIN):
        assert split in [TRAIN, VALID, TEST], f'Split must be either {TRAIN}, {VALID}, or {TEST}'
        self._target_df = self.df.query(f'split=="{split}"').reset_index(drop=True)

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        english_sentence, french_sentence = row[ENGLISH], row[FRENCH]
        english_sent_vec = self.vectorizer.vectorize_sentence(english_sentence, language=ENGLISH)
        french_sentence_vec = self.vectorizer.vectorize_sentence(french_sentence, language=FRENCH)

        # skip sos token in encoder input
        english_sent_vec = english_sent_vec[1:]
        french_sentence_vec = french_sentence_vec[:-1]
        return {
            ENCODER_INPUT: english_sent_vec,
            DECODER_INPUT: french_sentence_vec
        }

    def __len__(self):
        return len(self._target_df)


def generate_batches(
    dataset,
    batch_size,
    device='cpu'
    ):
    """
    This generator wraps the DataLoader class to build tensors out of the
    raw data and send them to the desired device
    """
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
        )

    for data_dict in data_loader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = tensor.to(device)
        yield out_data_dict
