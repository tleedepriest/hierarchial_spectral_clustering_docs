"""
This script will apply spectral clustering to the keywords.csv file
on the keywords previously extracted from the documents.
"""
import random
from math import floor, sqrt
import pandas as pd
import numpy as np

import optparse

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

import hdbscan
from sklearn.cluster import KMeans
# SpectralClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.decomposition import TruncatedSVD

def get_trigrams(df):
    """
    Return keywrods from keywords.csv
    """
    clean = df['clean_text'].astype(str).tolist()
    clean = [words.split() for words in clean]
    return clean

def get_keywords(df):
    """
    Return keywrods from keywords.csv
    """
    keywords = df['keywords'].astype(str).tolist()
    keywords = [key.split() for key in keywords]
    return keywords

def get_embedding_dict(path_to_glove):
    """
    Parameters
    ------------
    path_to_glove: str
        path to text file containing pretrained embeddings

    Returns
    ------------
    emb_dict: Dict[str, np.asarray[float32]]
        dictionary assigning words to vector
        embeddings
    """
    print(path_to_glove)
    emb_dict = {}
    with open(path_to_glove, 'r') as fh:
        for line in fh:
            split_line = line.split()
            word = split_line[0]
            numbers = split_line[1:]
            vector = np.asarray(numbers, 'float32')
            emb_dict[word] = vector
    return emb_dict

def train_embeddings(df):
    """
    Train the vector embeddings
    """
    #emb_dict = get_embedding_dict('glove.6B/glove.6B.100d.txt')
    path = "word2vec.model"
    clean_text = get_trigrams(df)
    model = Word2Vec(sentences=clean_text,
                     vector_size=50, window=2, min_count=5, workers=4)
    model.save(path)

def load_embeddings(df):
    path = "word2vec.model"
    model = Word2Vec.load("word2vec.model")
    df = pd.read_csv('keywords_with_clean_txt.csv')
    keywords = get_trigrams(df)
    doc_embeddings = np.zeros((len(keywords), 50))
    for doc_num, keyword in enumerate(keywords):
        doc_embedding = []
        for word in keyword:
            try:
                word_vector = model.wv[word]
            except KeyError:
                # create empty array for words not in model.
                word_vector = np.zeros(50)
            doc_embedding.append(word_vector)
        doc_embedding = np.array(doc_embedding)
        doc_embedding = doc_embedding.mean()
        doc_embeddings[doc_num, :] = doc_embedding
    number_clusters = floor(sqrt(len(keywords)))
    print(number_clusters)

    # clustering of doc embeddings, which are just mean of word embeddings
    # seem to result in clustering which resembles more or less data in
    # a straight line?
#   clustering = hdbscan.HDBSCAN(min_cluster_size=50).fit(doc_embeddings)
#   labels = clustering.labels_
    clusters = KMeans(n_clusters=20).fit_predict(doc_embeddings)
    df['labels'] = clusters.tolist()
    df.to_csv('test.csv')
    clusters = KMeans(n_clusters=20).fit_transform(doc_embeddings)
#   # Number of clusters in labels, ignoring noise if present.
    #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #print('Estimated number of clusters: %d' % n_clusters_)
    projection = TSNE(n_components=2, perplexity=60).fit_transform(doc_embeddings)
    plot_kwds = {'alpha' : 0.25, 's' : 10, 'linewidths':0}
    plt.scatter(*projection.T)
    plt.savefig('doc_embeddings_kemans.png')
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('--tfidf', default=True, dest='with_tfidf')
    df = pd.read_csv('keywords_with_clean_txt.csv')
    df = df.dropna(subset='clean_text')
    # shuffle rows in dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    df_train =df.loc[:10000, :]
    (options, args) = parser.parse_args()
    with_tfidf = options.with_tfidf
    # dataset is evenly distributed, hopefully fine split.
    train_labels = df_train["doc_path"].astype(str).to_list()
    train_text = df_train["clean_text"].astype(str).to_list()
    print(train_text)
    if with_tfidf:
        tfidf_vec = TfidfVectorizer(input='content')
        X_train = tfidf_vec.fit_transform(train_text)
        svd = TruncatedSVD(n_components=20)
        X_reduced_dimensions = svd.fit_transform(X_train)
        print(X_reduced_dimensions)
    else:
        train_embeddings(df)
        load_embeddings(df)

    clusters = KMeans(n_clusters=20).fit_predict(X_train)
    df_train['labels'] = clusters.tolist()
    df_train.to_csv('test.csv')
    fig, axes = plt.subplots(nrows=4, ncols=5)
    flat_axes = axes.flatten()
    name = "tab20b"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors  # type: list
    for i in range(0, 19):
        flat_axes[i].set_prop_cycle(color=colors)
        flat_axes[i].scatter(X_reduced_dimensions[:, i], X_reduced_dimensions[:, i+1], c=df_train['labels'].astype(float))
    plt.savefig("dim_one_and_two.png")
    projection = TSNE(n_components=2, perplexity=2).fit_transform(X_reduced_dimensions)
    plot_kwds = {'alpha' : 0.25, 's' : 10, 'linewidths':0}

    fig, axes = plt.subplots()
    plt.scatter(*projection.T, c=df_train['labels'].astype(float))
    plt.savefig('doc_embeddings_kemans.png')
