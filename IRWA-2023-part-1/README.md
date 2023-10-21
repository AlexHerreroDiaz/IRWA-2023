# Text Processing and Exploratory Data Analysis

## Part 1 - Text Processing

### 1. Document Pre-process

This section covers the preprocessing of text data. It includes:

- Mounting Google Drive using the Colab library.
- Loading essential Python packages such as NLTK for natural language processing and text analysis.
- Loading the dataset from the csv files at the `\IRWA_data_2023` directory
- Defining a function, `split_hashtag_words`, to handle hashtags with multiple words.
- Creating a function, `build_terms`, to preprocess text by removing stop words, stemming, transforming to lowercase, and returning a list of tokens.
- A function, `get_data_ids`, retrieves document IDs for each tweet.
- Creating a dictionary of tweets with various attributes like text, date, hashtags, likes, retweets, and tokens.

### 2. Exploratory Data Analysis

In this section, you perform an exploratory data analysis, including:

- Determining the vocabulary size of the dataset.
- Calculating the average token length for the entire dataset.
- Identifying the top 10 stemmed words with the most occurrences in the data.
- Listing the top 10 most liked and retweeted tweets.
- Generating a WordCloud visualization for the most frequent tokens.
- Utilizing the spaCy NLP model to extract the top 10 most frequent entities and their types.

The code is well-documented and provides examples to illustrate each step. This notebook serves as a comprehensive guide for text processing and Exploratory Data Analysis.
