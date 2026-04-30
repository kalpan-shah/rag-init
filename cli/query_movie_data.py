""" 
@file:          cli/keyword_search_cli.py
@description:   Keyword search CLI for finding movies
@date:          30 April 2026
@last edited:   30 April 2026
@author:        Kalpan Shah
@version:       1.0.0 
"""
from typing import List
import os
import json
import string

# Assuming running from the project root, adjust path accordingly later if needed
movie_data_json = os.path.join(os.getcwd(),'data', 'movies.json')

def load_data(file_path: str) -> dict:
    """
        Helper function to just load the json and return the data dict
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

movie_data: List[dict] = load_data(movie_data_json)['movies']

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
    _tokens = list(filter(str.strip, _tokens))
    return _tokens




def search_movies(query: str) -> List[dict]:
    """
        V1 - Search for movies if the movie name contains the query string 
            Text Processing:
                a. Case insensitive search
                b. punctuation removal
                c. tokenization
    """
    results = []
    _query = preprocess_text(query)
    _query_tokens = get_tokens(_query)

    for movie in movie_data:
        # Check if Valid movie dict
        if not isinstance(movie, dict) or 'title' not in movie:
            continue

        # text preprocessing
        _movie_title = preprocess_text(movie['title'])
        # _movie_title_tokens = get_tokens(_movie_title)

        _token_match = any(token in _movie_title for token  in _query_tokens)
        if _token_match:
            results.append(movie)

    return results
