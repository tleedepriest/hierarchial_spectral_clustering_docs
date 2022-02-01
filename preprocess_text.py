"""
This script reads in the documents and performs pre-processing on the documents
"""
import pandas as pd
from iLDA_wrapper import iLDA
from gensim.parsing.preprocessing import split_on_space, preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_short, remove_stopwords
from gensim.parsing.preprocessing import preprocess_string
from gensim.models import Phrases

from utils import remove_emails, remove_in_article, remove_names, remove_arrow
def main():
    # first retrieve text, tokens, dictionary, and corpus
    string_filters = [remove_emails,
                      remove_in_article,
                      remove_names,
                      remove_arrow,
                      lambda x: x.lower(),
                      strip_tags,
                      strip_punctuation,
                      strip_multiple_whitespaces,
                      remove_stopwords,
                      strip_short]
    ilda = iLDA(docs_dir='20news_home', string_filters=string_filters)
    ilda.set_attributes(with_bigrams=True)
    tokens = ilda.tokens
    doc_paths = ilda.doc_paths
    tokens = ['  '.join(tok) for tok in tokens]
    df = pd.DataFrame()
    df["clean_text"] = tokens
    df["doc_path"] = [str(doc_path) for doc_path in doc_paths]
    keywords = pd.read_csv('keywords.csv')
    new_df = keywords.merge(df, on="doc_path")
    new_df.to_csv('keywords_with_clean_txt.csv', index=False)

if __name__ == '__main__':
    main()
