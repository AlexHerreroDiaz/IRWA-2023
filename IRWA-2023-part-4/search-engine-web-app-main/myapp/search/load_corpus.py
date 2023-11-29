import pandas as pd
import json
import csv
import os
import pickle

from myapp.core.utils import load_json_file
from myapp.search.objects import Document
from myapp.search.algorithms import build_terms
from collections import defaultdict
from array import array
import math
import numpy as np


_corpus = {}


def load_corpus(path) -> [Document]:
    """
    Load file and transform to dictionary with each document as an object for easier treatment when needed for displaying
     in results, stats, etc.
    :param path:
    :return:
    """
    df = _load_corpus_as_dataframe(path)
    df.apply(_row_to_doc_dict, axis=1)
    return _corpus


def _load_corpus_as_dataframe(path):
    """
    Load documents corpus from file in 'path'
    :return:
    """
    json_data = load_json_file(path)
    tweets_df = _load_tweets_as_dataframe(json_data)
    _clean_hashtags_and_urls(tweets_df)
    # Rename columns to obtain: Tweet | Username | Date | Hashtags | Likes | Retweets | Url | Language
    corpus = tweets_df.rename(
        columns={"id": "Id", "full_text": "Tweet", "screen_name": "Username", "created_at": "Date",
                 "favorite_count": "Likes",
                 "retweet_count": "Retweets", "lang": "Language"})

    # select only interesting columns
    filter_columns = ["Id", "Tweet", "Username", "Date", "Hashtags", "Likes", "Retweets", "Url", "Language"]
    corpus = corpus[filter_columns]
    return corpus


def _load_tweets_as_dataframe(json_data):
    data = pd.DataFrame(json_data).transpose()
    # parse entities as new columns
    data = pd.concat([data.drop(['entities'], axis=1), data['entities'].apply(pd.Series)], axis=1)
    # parse user data as new columns and rename some columns to prevent duplicate column names
    data = pd.concat([data.drop(['user'], axis=1), data['user'].apply(pd.Series).rename(
        columns={"created_at": "user_created_at", "id": "user_id", "id_str": "user_id_str", "lang": "user_lang"})],
                     axis=1)
    return data


def _build_tags(row):
    tags = []
    # for ht in row["hashtags"]:
    #     tags.append(ht["text"])
    for ht in row:
        tags.append(ht["text"])
    return tags


def _build_url(row):
    url = ""
    try:
        url = row["entities"]["url"]["urls"][0]["url"]  # tweet URL
    except:
        try:
            url = row["retweeted_status"]["extended_tweet"]["entities"]["media"][0]["url"]  # Retweeted
        except:
            url = ""
    return url


def _clean_hashtags_and_urls(df):
    df["Hashtags"] = df["hashtags"].apply(_build_tags)
    df["Url"] = df.apply(lambda row: _build_url(row), axis=1)
    # df["Url"] = "TODO: get url from json"
    df.drop(columns=["entities"], axis=1, inplace=True)


def load_tweets_as_dataframe2(json_data):
    """Load json into a dataframe

    Parameters:
    path (string): the file path

    Returns:
    DataFrame: a Panda DataFrame containing the tweet content in columns
    """
    # Load the JSON as a Dictionary
    tweets_dictionary = json_data.items()
    # Load the Dictionary into a DataFrame.
    dataframe = pd.DataFrame(tweets_dictionary)
    # remove first column that just has indices as strings: '0', '1', etc.
    dataframe.drop(dataframe.columns[0], axis=1, inplace=True)
    return dataframe


def load_tweets_as_dataframe3(json_data):
    """Load json data into a dataframe

    Parameters:
    json_data (string): the json object

    Returns:
    DataFrame: a Panda DataFrame containing the tweet content in columns
    """

    # Load the JSON object into a DataFrame.
    dataframe = pd.DataFrame(json_data).transpose()

    # select only interesting columns
    filter_columns = ["id", "full_text", "created_at", "entities", "retweet_count", "favorite_count", "lang"]
    dataframe = dataframe[filter_columns]
    return dataframe


def _row_to_doc_dict(row: pd.Series):
    _corpus[row['Id']] = Document(row['Id'], row['Tweet'][0:100], row['Tweet'], row['Date'], row['Likes'],
                                  row['Retweets'],
                                  row['Url'], row['Hashtags'])


def get_data_ids(file_path):
  id_to_doc = defaultdict(list)

  with open(file_path, 'r') as file:
    csv_reader = csv.reader(file, delimiter='\t')

    for row in csv_reader:
      id_to_doc[row[1]] = row[0]
  return id_to_doc

def array_to_list(arr):
    """Converts array objects to lists."""
    return arr.tolist() if isinstance(arr, array) else arr


def load_tweets_as_documents(path, num_documents=4000):
    """
    Load tweets from a file and convert them into Document objects.

    Arguments:
    file_path -- path to the file containing tweet data

    Returns:
    corpus -- a dictionary with tweet IDs as keys and Document objects as values
    """

    file_path = path + "/data/Rus_Ukr_war_data.json"
    ids_path = path + "/data/Rus_Ukr_war_data_ids.csv"
    
    corpus_file = path + "/var/corpus.pkl"
    index_file = path + "/var/index.pkl"
    tf_file = path + "/var/tf.pkl"
    df_file = path + "/var/df.pkl"
    idf_file = path + "/var/idf.pkl"
    title_index_file = path + "/var/title_index.pkl"
    data_ids_file = path + "/var/data_ids_dict.pkl"

    # Check if all files exist to avoid re-computation
    if all(os.path.exists(file) for file in [corpus_file, index_file, tf_file, df_file, idf_file, title_index_file, data_ids_file]):
        # Load existing files if they all exist
        with open(corpus_file, 'rb') as file:
            corpus = pickle.load(file)
        with open(index_file, 'rb') as file:
            index = pickle.load(file)
        with open(tf_file, 'rb') as file:
            tf = pickle.load(file)
        with open(df_file, 'rb') as file:
            df = pickle.load(file)
        with open(idf_file, 'rb') as file:
            idf = pickle.load(file)
        with open(title_index_file, 'rb') as file:
            title_index = pickle.load(file)
        with open(data_ids_file, 'rb') as file:
            data_ids_dict = pickle.load(file)
        return corpus, index, tf, df, idf, title_index, data_ids_dict
    
    corpus = {}
    index = defaultdict(list)
    tf = defaultdict(list)  #term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  #document frequencies of terms in the corpus
    title_index = defaultdict(str)
    idf = defaultdict(float)

    data_ids_dict = get_data_ids(ids_path)

    # Your code to load and process tweets here
    with open(file_path, 'r') as file:
        count = 0
        for line in file:
            tweet_data = json.loads(line)
            # Extract necessary fields from tweet_data
            tweet_id = tweet_data['id']
            tweet_id_str = tweet_data['id_str']
            title = tweet_data['full_text'][:100]   # You may adjust this based on your desired 'title' field
            description = tweet_data['full_text'] # Adjust the description field as needed
            doc_date = tweet_data['created_at']
            likes = tweet_data['favorite_count']
            retweets = tweet_data['retweet_count']
            url = f"https://twitter.com/twitter_username/status/{tweet_data['id_str']}"
            hashtags = [hashtag['text'] for hashtag in tweet_data['entities']['hashtags']]

            terms, line = build_terms(description)

            # Create Document object
            doc = Document(tweet_id, title, description, doc_date, likes, retweets, url, hashtags)
            corpus[tweet_id] = doc

            title_index[data_ids_dict[tweet_id_str]] = description

            # Create the index for the current tweet
            current_tweet_index = {}

            for position, term in enumerate(terms): # terms contains page_title + page_text. Loop over all terms
                try:
                    # if the term is already in the index for the current page (current_page_index)
                    # append the position to the corresponding list
                    current_tweet_index[term][1].append(position)
                except:
                    # Add the new term as dict key and initialize the array of positions and add the position
                    current_tweet_index[term]=[data_ids_dict[tweet_id_str], array('I',[position])] #'I' indicates unsigned int (int in Python)

            # normalize term frequencies
            # Compute the denominator to normalize term frequencies (formula 2 above)
            # norm is the same for all terms of a document.
            norm = 0
            for term, posting in current_tweet_index.items():
                # posting will contain the list of positions for current term in current document.
                # posting ==> [current_doc, [list of positions]]
                # you can use it to infer the frequency of current term.
                norm += len(posting[1]) ** 2
            norm = math.sqrt(norm)

            #calculate the tf(dividing the term frequency by the above computed norm) and df weights
            for term, posting in current_tweet_index.items():
                # append the tf for current term (tf = term frequency in current doc/norm)
                tf[term].append(np.round(len(posting[1]) / norm, 4)) ## SEE formula (1) above
                #increment the document frequency of current term (number of documents containing the current term)
                df[term] += 1 # increment DF for current term

            #merge the current page index with the main index
            for term_page, posting_page in current_tweet_index.items():
                index[term_page].append(posting_page)

            # Compute IDF following the formula (3) above. HINT: use np.log
            for term in df:
                idf[term] = np.round(np.log(float(num_documents / df[term])), 4)
            count +=1
            if(count % 500 == 0):
                print(f'{count}/{num_documents} lines processed')
            if(count>num_documents):
                break
    
    # Save the variables using pickle
    with open(corpus_file, 'wb') as file:
        pickle.dump(corpus, file)
    with open(index_file, 'wb') as file:
        pickle.dump(index, file)
    with open(tf_file, 'wb') as file:
        pickle.dump(tf, file)
    with open(df_file, 'wb') as file:
        pickle.dump(df, file)
    with open(idf_file, 'wb') as file:
        pickle.dump(idf, file)
    with open(title_index_file, 'wb') as file:
        pickle.dump(title_index, file)
    with open(data_ids_file, 'wb') as file:
        pickle.dump(data_ids_dict, file)

    return corpus, index, tf, df, idf, title_index, data_ids_dict