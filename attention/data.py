"""This module contains data management logic for NMT tasks. """
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from attention.vectorizer import NMTVectorizer
from attention.constants import (
    SPLIT,
    TRAIN,
    VALID,
    TEST,
    ENGLISH,
    FRENCH
    )


class NMTDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for a Machine Translation task.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        vectorizer: NMTVectorizer
        ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.vectorizer = vectorizer
        self.set_split()

    @classmethod
    def from_dataframe(cls, dataframe):
        """
        Instantiate the class from a pandas dataframe.
        """
        vectorizer = NMTVectorizer.from_df(dataframe)
        return cls(dataframe, vectorizer)

    def set_split(self, split: str=TRAIN):
        """
        Set the target dataframe to one of three splits: train, eval, test
        """
        assert split in [TRAIN, VALID, TEST], f'Split must be either {TRAIN}, {VALID}, or {TEST}'
        self._target_df = self.dataframe.query(f'split=="{split}"').reset_index(drop=True)

    def __getitem__(self, index):
        """
        Retrieve a single record from the target dataset.
        """
        row = self._target_df.iloc[index]
        english_sentence, french_sentence = row[ENGLISH], row[FRENCH]
        return self.vectorizer.vectorize(english_sentence, french_sentence)

    def __len__(self):
        return len(self._target_df)


def generate_batches(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    device
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


def generate_nmt_batches(dataset, batch_size, device, drop_last=True, shuffle=True):
    """
    Creates a generator object that generates batches from a pytorch dataset.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )

    for data_dict in dataloader:
        lengths = data_dict['source_length'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()

        out_data_dict = {}
        for name in data_dict.keys():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict


def load_sentences_dataframe(file_path: str)-> pd.DataFrame:
    """
    Loads a pandas dataframe.
    """
    dataframe = pd.read_csv(file_path, sep='\t', header=None)
    dataframe.columns = ['english', 'french']
    return dataframe


def assign_rows_to_split(
    dataframe: pd.DataFrame,
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
        dataframe,
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
