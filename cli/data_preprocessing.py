""" 
@file:          cli/data_preprocessing.py
@description:   Helper functions for preprocessing movie data
@date:          30 April 2026
@last edited:   30 April 2026
@author:        Kalpan Shah
@version:       1.0.0 
"""
from typing import List
import string
from nltk.stem import PorterStemmer

from data_handler import stop_words

stemmer = PorterStemmer()


def preprocess_text(text: str) -> str:
    """
        Helper function to preprocess text for better search results
        a. Case insensitive search
        b. punctuation removal
    """
    # a. Case insensitive search
    text = text.lower()

    # b. punctuation removal
    # Define a translation table that maps punctuation characters to None
    tab = str.maketrans('', '', string.punctuation)  # mapping for removing punctuation
    text = text.translate(tab)  # mapping applied using translate method

    return text

def get_tokens(text: str) -> List[str]:
    """
        Helper function to tokenize text for better search results
    """
    _tokens = text.split()
    # remove empty tokens if any
    _tokens = [token for token in _tokens if token.strip()]

    # remove stop words
    _tokens = [token for token in _tokens if token not in stop_words]

    # stemming - reduct to root form
    _tokens = [stemmer.stem(token) for token in _tokens]

    return _tokens

