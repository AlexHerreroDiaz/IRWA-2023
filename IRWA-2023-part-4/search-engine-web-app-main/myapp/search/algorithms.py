from collections import defaultdict
import collections
import math
from numpy import linalg as la
import numpy as np
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# Assume that tf, idf, index, and title_index are provided or accessible
# You might need to pass these parameters into the function if they are not globally available

def split_hashtag_words(text):
  tokens = re.findall(r'\w+|#\w+', text)

  result_tokens = []

  for token in tokens:
      if token.startswith('#'):
          # Remove the '#' symbol and split the hashtag into words by uppercase letters
          hashtag = token[1:]
          hashtag_words = re.findall(r'[A-Z][a-z]*', hashtag)

          # Check if all words in the hashtag are in uppercase, then treat it as a single word
          if all(word.isupper() for word in hashtag_words):
              result_tokens.append(hashtag)
          else:
              result_tokens.extend(hashtag_words)
      else:
          result_tokens.append(token)

  return result_tokens

def build_terms(line, stemming=True):
    """
    Preprocess the text by removing stop words, stemming,
    transforming to lowercase, and returning the tokens of the text.

    Arguments:
    line -- string (text) to be preprocessed

    Returns:
    terms -- a list of tokens corresponding to the input text after preprocessing
    """
    # Remove URLs
    line = re.sub(r'http\S+', '', line)

    temp_terms = split_hashtag_words(line)
    line = " ".join(temp_terms)

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    # Transform to lowercase
    line = line.lower()

    # Remove punctuation, emojis, symbols, numbers, and strings starting with '#' and '@'
    line = re.sub(r'@\w+', '', line)
    line = re.sub(r'#\w+', '', line)
    line = re.sub(r'[^\w\s]', '', line)
    line = re.sub(r'[\d]', '', line)

    # Tokenize the text to get a list of terms
    # Initialize the TweetTokenizer
    tokenizer = TweetTokenizer()

    # Tokenize the tweet into words
    terms = split_hashtag_words(line)

    # Remove stopwords and perform stemming
    if (stemming):
      terms = [stemmer.stem(word) for word in terms if word not in stop_words]

    # Join the words back into a sentence
    line = " ".join(terms)

    return terms, line

def get_keys(dictionary, value):
    return [key for key, val in dictionary.items() if val == value]

def rank_documents(terms, docs, index_d, idf, tf, title_index):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    title_index -- mapping between page id and page title

    Returns:
    Print the list of ranked documents
    """

    # I'm interested only on the element of the docVector corresponding to the query terms
    # The remaining elements would become 0 when multiplied to the query_vector
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})
    #HINT: use when computing tf for query_vector

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index_d:
            continue

        # check how to vectorize the query
        # query_vector[termIndex]=idf[term]  # original
        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index_d[term]):
            # Example of [doc_index, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # 1 (33, array('I', [26, 33, 57, 71, 87, 104, 109]))
            # term is in doc 26 in positions 1,4, .....
            # term is in doc 33 in positions 26,33, .....

            #tf[term][0] will contain the tf of the term "term" in the doc 26
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]  # check if multiply for idf

    # Calculate the score of each doc
    # compute the cosine similarity between queyVector and each docVector:
    # HINT: you can use the dot product because in case of normalized vectors it corresponds to the cosine similarity
    # see np.dot

    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    #print document titles instead if document id's
    #result_docs=[ title_index[x] for x in result_docs ]
    if len(result_docs) == 0:
        print("No results found")
        result_docs = []
        doc_scores = []
    return result_docs, doc_scores

def search_in_corpus(query, index_d, tf, df, idf, title_index, data_ids):
    # Implementation of search_in_corpus function using provided functions and data structures

    # 1. Preprocess the query terms
    query_terms, _ = build_terms(query)

    # 2. Perform search using TF-IDF scoring
    relevant_docs = set()
    for term in query_terms:
        if term in index_d:
            term_docs = [posting[0] for posting in index_d[term]]
            if not relevant_docs:
                relevant_docs = set(term_docs)
            else:
                relevant_docs = relevant_docs.intersection(term_docs)

    relevant_docs = list(relevant_docs)

    # 3. Rank the relevant documents using TF-IDF
    ranked_docs, doc_scores = rank_documents(query_terms, relevant_docs, index_d, idf, tf, title_index)
    integers_list = [int(x.split('_')[1]) for x in ranked_docs if x.split('_')[1].isdigit()]
    integers_list = [x - 1 for x in integers_list]
    return integers_list, doc_scores  # Return the ranked list of relevant documents based on the query
