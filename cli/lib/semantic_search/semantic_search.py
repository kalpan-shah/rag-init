"""
@file:          cli/lib/semantic_search/semantic_search.py
@description:   Semantic search implementation for finding movies
@date:          08 May 2026
@last edited:   08 May 2026
@author:        Kalpan Shah
@version:       1.0.0
"""
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    """
        Semantic search implementation using sentence transformers.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self._max_sequence_length = 0
        self.max_sequence_length = self.model.get_max_seq_length()

    def encode(self, text: str):
        """
            Encode the input text into a dense vector representation.
            Args:
                text (str): Input text to encode
            Returns:
                np.ndarray: Dense vector representation of the input text
        """
        return self.model.encode(text)

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

def verify_model():
    _searcher = SemanticSearch()
    print(f"Model loaded: {_searcher.model}")
    print(f"Max sequence length: {_searcher.max_sequence_length}")
