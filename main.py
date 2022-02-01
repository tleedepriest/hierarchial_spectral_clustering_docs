"""
This is the main script that runs the luigi pipeline
"""
import luigi
from extract_keywords import main
from train_embeddings import train_embeddings

class ExtractKeyWords(luigi.Task):
    """
    Uses the YAKE package to extract keywords
    """

    def requires(self):
        return None

    def output(self):
        return luigi.LocalTarget("keywords.csv")

    def run(self):
        extract_keywords.main()

class TrainEmbeddings(luigi.Task):
    """
    Train word2vec vector embeddings
    on the keywords of the model
    """
    def requires(self):
        return ExtractKeyWords()

    def output(self):
        return luigi.LocalTarget('word2vec.model')

    def run(self):
        train_embeddings()

if __name__ == '__main__':
    luigi.run()
