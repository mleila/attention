import spacy
import fasttext

from attention.constants import ENGLISH, FRENCH


def create_fasttext_embeddings(vectorizer, download=False):
    if download:
        fasttext.util.download_model('en', if_exists='ignore')
        fasttext.util.download_model('fr', if_exists='ignore')

    english_model = fasttext.load_model('cc.en.300.bin')
    french_model = fasttext.load_model('cc.fr.300.bin')

    english_embedding_matrix = vectorizer.build_embedding_matrix_from_fasttext(english_model, lang=ENGLISH)
    french_embedding_matrix = vectorizer.build_embedding_matrix_from_fasttext(french_model, lang=FRENCH)
    return english_embedding_matrix, french_embedding_matrix


def create_spacy_embeddings(vectorizer, download=False):
    if download:
        message = """
        #!python -m spacy download fr_core_news_md
        #!python -m spacy download en_core_web_sm
        """
        print(message)
        return

    english_model = spacy.load('en_core_web_sm')
    french_model = spacy.load('fr_core_news_md')

    english_embedding_matrix = vectorizer.build_embedding_matrix_from_spacy(english_model, lang=ENGLISH)
    french_embedding_matrix = vectorizer.build_embedding_matrix_from_spacy(french_model, lang=FRENCH)
    return english_embedding_matrix, french_embedding_matrix