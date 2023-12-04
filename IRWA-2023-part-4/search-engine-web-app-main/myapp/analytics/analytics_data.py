import json
import random

from myapp.search.algorithms import build_terms


class AnalyticsData:
    """
    An in-memory persistence object.
    Declare more variables to hold analytics tables.
    """
    def __init__(self):
        self.fact_clicks = dict([])
        self.fact_two = dict([])
        self.fact_three = dict([])
        self.query_terms = {}

    def save_query_terms(self, terms: str) -> int:
        # Tokenize the input terms using NLTK's word_tokenize or built_terms function
        tokens, _ = build_terms(terms)  # You can also use build_terms function here
        
        # Store individual tokens in the query_terms dictionary
        for token in tokens:
            # Convert token to lowercase for uniformity
            token = token.lower()
            
            # Check if the token exists in query_terms, if yes, increment count, else add it
            if token in self.query_terms:
                self.query_terms[token] += 1
            else:
                self.query_terms[token] = 1
        
        return random.randint(0, 100000)

class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return {
            'doc_id': self.doc_id,
            'description': self.description,
            'counter': self.counter
        }
    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
