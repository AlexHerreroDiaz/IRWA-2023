# Indexing and Evaluation of Tweets - Notebook Overview

This notebook implements a system to index and evaluate a collection of tweets related to the Russia-Ukraine conflict. The system consists of several components: preprocessing the tweet data, building an inverted index using TF-IDF, querying the index, and evaluating the performance using various metrics.

## Group Members

- **Guillem Gauchia Torres** - 240215 - u186410
- **Àlex Herrero Díaz** - 240799 - u186402
- **Adrià Julià Parada** - 242195 - u188319

## Overview

The Notebook is structured as follows:

### 1. Data Pre-processing

- **Text Preprocessing**: Utilizes techniques such as removing stopwords, stemming, and tokenization for the tweets.

### 2. Indexing and TF-IDF Computation

- **Building the Inverted Index**: Constructs an inverted index using TF-IDF techniques to facilitate efficient querying of tweets.

### 3. Querying the Index

- **Ranking Documents**: Ranks the documents based on a search query using the TF-IDF weights.

### 4. Evaluation of Search Results

- **Evaluation Metrics**: Computes several evaluation metrics including Precision, Recall, F1-Score, Mean Average Precision, Mean Reciprocal Rank, and Normalized Discounted Cumulative Gain.

### 5. Visualization

- **T-SNE Visualization of Tweets**: Provides a 2D visualization of tweets based on their TF-IDF vectors using t-SNE.

## Files and Data

- **Input Data**: Loading tweet data from a JSON file.
- **Evaluation Ground Truth**: Evaluation file to compare search results and ground truth.

## How to Use the Notebook

- **Load Data**: Mount Google Drive and load the necessary data for indexing and evaluation.
- **Run Functions**: Execute functions for preprocessing, index creation, querying, and evaluation.
- **Evaluate Performance**: Check evaluation metrics such as Precision, Recall, and more.