import pandas as pd


def load_sentences_dataframe(fp):
    df = pd.read_csv(fp, sep='\t', header=None)
    df.columns = ['english', 'french']
    return df
