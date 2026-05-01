"""
@file:          cli/inverted_index.py
@description:   Inverted index implementation for efficient keyword search
@date:          29 April 2026 
@last edited:   01 April 2026
@author:        Kalpan Shah
@version:       1.0.0 
"""
import os
from typing import Dict, List, Set
import pickle
from data_preprocessing import get_tokens, preprocess_text
from data_handler import movie_data, index_file_path, docmap_file_path

class InvertedIndex:
    """
        Inverted index data structure for efficient keyword search.
    """
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, object] = {} 
    
    def __add_document(self, doc_id: int, text: str) -> None:
        """
            Add a document to the inverted index.
            Args:
                doc_id (int): Unique identifier for the document
                text (str): Text content of the document
        """
        # preprocess text
        text = preprocess_text(text)
        # a. get tokens
        tokens = get_tokens(text)
        # b. update index
        for token in tokens:
            # create a new set for the token if it doesn't exist
            if token not in self.index:
                self.index[token] = set()
            # update the set of document IDs for the token
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> List[int]:
        """
            Retrieve document IDs associated with a given term - token.
            Args:
                term (str): The search term
            Returns:
                List[int]: A list of document IDs containing the term
        """
        _doc_ids = list(self.index.get(term.lower(), set()))
        _doc_ids.sort()
        return _doc_ids
    
    def build(self) -> None:
        """
            Build the inverted index from the movies data.
        """
        for movie in movie_data:
            _in_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie['id'], _in_text)

            # movie.update({'full_text': _in_text}) 
            # self.docmap[movie['id']] = movie
        
        # Sort the document IDs for each token in the index for consistent retrieval
        for token in self.index:
            self.index[token] = set(sorted(self.index[token]))

        # save the index and docmap to disk
        self.save()
        
    def save(self) -> None:
        """
            Save the inverted index and document mapping to disk using pickle.
        """
        with open(index_file_path, 'wb') as i_f:
            pickle.dump(self.index, i_f)
        
        with open(docmap_file_path, 'wb') as d_f:
            pickle.dump(self.docmap, d_f)

    def load(self) -> None:
        """
            Load the inverted index and document mapping from disk using pickle.
        """

        if not os.path.exists(index_file_path) or not os.path.exists(docmap_file_path):
            raise FileNotFoundError("Inverted index files not found. Please build the index first.")

        with open(index_file_path, 'rb') as f:
            self.index = pickle.load(f)
        
        with open(docmap_file_path, 'rb') as f:
            self.docmap = pickle.load(f)
