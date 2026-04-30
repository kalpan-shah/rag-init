"""
@file:          cli/data_handler.py
@description:   Helper functions for loading movie data and stop words
@date:          30 April 2026
@last edited:   30 April 2026
@author:        Kalpan Shah
@version:       1.0.0 
"""
import os
import json 
from typing import List, Set


# Assuming running from the project root, adjust path accordingly later if needed
movie_data_json = os.path.join(os.getcwd(),'data', 'movies.json')
stop_word_file = os.path.join(os.getcwd(), 'data', 'stopwords.txt')

def load_data(file_path: str) -> dict:
    """
        Helper function to just load the json and return the data dict
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_stop_words(file_path: str) -> Set[str]:
    """
        Helper function to load stop words from a text file and return a list of stop words
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        _words = f.read().splitlines()
    return set(_words)


stop_words: Set[str] = load_stop_words(stop_word_file)
movie_data: List[dict] = load_data(movie_data_json)['movies']
