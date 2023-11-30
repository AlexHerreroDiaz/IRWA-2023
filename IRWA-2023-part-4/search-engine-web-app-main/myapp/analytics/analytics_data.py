import json
import random


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
        # Convert query terms to lower case
        terms_lower = terms.lower()

        # Store the lower-cased query terms and their counts in query_terms dictionary
        if terms_lower in self.query_terms:
            self.query_terms[terms_lower] += 1
        else:
            self.query_terms[terms_lower] = 1

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
