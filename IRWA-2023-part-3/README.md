# Ranking and Evaluation of Tweets - Notebook Overview

This notebook focuses on the ranking and evaluation of tweets related to the Russia-Ukraine conflict. The ranking is performed using two different approaches: **Social-Score + Cosine Similarity** and **Word2Vec + Cosine Similarity**. Additionally, there is a report of the remaining sections.

## Group Members
- Guillem Gauchia Torres - 240215 - u186410
- Àlex Herrero Díaz - 240799 - u186402
- Adrià Julià Parada - 242195 - u188319

## Overview

The notebook is structured into two main sections:

### 1. Social-Score + Cosine Similarity Ranking

In this section, a ranking method called "Social-Score" is implemented. This method considers social parameters such as Likes, Retweets, and Followers. The ranking is based on both Cosine Similarity and Euclidean Distance, providing a balanced approach. The top-ranked tweets are displayed along with their associated social parameters.

### 2. Word2Vec + Cosine Similarity Ranking

This section utilizes Word2Vec embeddings to represent tweets as vectors. The ranking is based on the cosine similarity between the query and the tweet vectors. Pre-trained Word2Vec models are used for vector representation. The top 20 ranked tweets for specific queries are printed with their original content.

## Files and Data

- **Input Data:** Tweet data is loaded from a JSON file.
- **Evaluation Ground Truth:** An evaluation file is used to compare search results with the ground truth.

## How to Use the Notebook

1. **Load Data:** Mount Google Drive and load the required data for indexing and evaluation.
2. **Run Functions:** Execute functions for preprocessing, indexing, querying, and evaluation.
3. **Evaluate Performance:** Check evaluation metrics such as Precision, Recall, and more.
