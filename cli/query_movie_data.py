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
    with open(file_path, 'r') as f:
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

def search_movies(query: str) -> List[dict]:
    """
        V1 - Search for movies if the movie name contains the query string 
            Text Processing:
                a. Case insensitive search
                b. punctuation removal
    """
    results = []
    for movie in movie_data:
        if not isinstance(movie, dict) and 'title' not in movie:
            continue
        _query = preprocess_text(query)
        _movie_title = preprocess_text(movie['title'])
        if _query in _movie_title:
            results.append(movie)
    return results
