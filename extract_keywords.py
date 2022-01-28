from pathlib import Path
import pandas as pd
import yake

def extract_key_words_helper(text):
    """
    List[Tuple[str, int]
    """
    kw_extractor = yake.KeywordExtractor()
    # results in a list of Tuples
    # List[Tuple(str, float)]
    # each tuple contains keyword and value to rank
    # keyword importance. sorted from most to least
    # important
    keywords = kw_extractor.extract_keywords(text)
    just_words = [keyword[0] for keyword in keywords]
    # create bigrams or trigrams for keywords with multiple
    # words or interior stop words
    just_words = ['_'.join(keyword.lower().split()) for keyword
                  in just_words]
    # consider only unique words
    just_words = list(set(just_words))
    # could add a lemma step here.
    # join the keywrods together seperated by space.
    keywords = '  '.join(just_words)
    return keywords


def main():
    docs_dir = '20news_home'
    doc_paths = [x for x in Path(docs_dir).glob("**/*") if x.is_file()][:10]
    data_dict = {}
    test = doc_paths[1].open('r').read()
    extract_key_words_helper(test)
    data_dict["doc_path"] = doc_paths
    data_dict["document"] = [Path(x).open('r').read() for x in doc_paths]
    df = pd.DataFrame().from_dict(data_dict)
    # have to instantiate the object everytime we call function
    # which is not ideal
    df["keywords"] = df["document"].map(extract_key_words_helper)
    df = df[["doc_path", "keywords"]]
    df.to_csv('test.csv', index=False)

if __name__ == '__main__':
    main()

