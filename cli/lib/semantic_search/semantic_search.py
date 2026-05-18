"""
@file:          cli/lib/semantic_search/semantic_search.py
@description:   Semantic search implementation for finding movies
@date:          08 May 2026
@last edited:   08 May 2026
@author:        Kalpan Shah
@version:       1.0.0
"""
import os
import pickle
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

embeddings_file = os.path.join(os.getcwd(), 'cache', 'movie_embeddings.npy')
docmap_file_path = os.path.join(os.getcwd(), 'cache', 'docmap.pkl')


class SemanticSearch:
    """
        Semantic search implementation using sentence transformers.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self._max_sequence_length = 0
        self.max_sequence_length = self.model.get_max_seq_length()

        self.embeddings = None
        self.documents = None
        self.document_map = None

    def encode(self, text: str):
        """
            Encode the input text into a dense vector representation.
            Args:
                text (str): Input text to encode
            Returns:
                List[np.ndarray]: Dense vector representation of the input text
        """
        return self.model.encode(text, show_progress_bar=True)

    @property
    def max_sequence_length(self) -> int:
        """
            Get the maximum sequence length supported by the model.
            Returns:
                int: Maximum sequence length
        """
        return self._max_sequence_length

    @max_sequence_length.setter
    def max_sequence_length(self, value: int) -> None:
        """
            Set the maximum sequence length for the model.
            Args:
                value (int): Maximum sequence length to set
        """
        self._max_sequence_length = value

    def generate_embedding(self, text: str):
        """
            Generate a dense vector embedding for the input text.
            Args:
                text (str): Input text to generate embedding using the encode method

            Returns:
                embedding (np.ndarray): Return the first Index of List returned by the encode method
        """
        if not text:
            raise ValueError("Input text cannot be empty")
        embeddings = self.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents: List[dict]):
        """
            Build dense vector embeddings for a list of documents.
            Args:
                documents (List[dict]): List of documents to build embeddings for
        """
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        texts = ["{}: {}".format(doc['title'], doc['description']) for doc in documents]
        self.embeddings = self.encode(texts)

        np.save(embeddings_file, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: List[dict]):
        """
            Load existing embeddings from disk or create new embeddings if they don't exist.
            Args:
                documents (List[dict]): List of documents to build embeddings for if they don't exist
        """
        if os.path.exists(embeddings_file) and self.embeddings is None:
            self.embeddings = np.load(embeddings_file)
        elif self.embeddings is None:
            self.build_embeddings(documents)
        
        # Load documents and document map if not already loaded
        self.documents = documents
        with open(docmap_file_path, 'rb') as d_f:
            self.document_map = pickle.load(d_f)