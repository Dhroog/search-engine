import pandas as pd
from modules import analyez


class DataSet:
    def __init__(self, path_docs, path_queries, path_grels):
        self.docs = pd.read_csv(path_docs)
        self.queries = pd.read_csv(path_queries)
        self.qrels = pd.read_csv(path_grels)

    def set_docs(self, path):
        self.docs = pd.read_csv(path)

    def set_queries(self, path):
        self.queries = pd.read_csv(path)

    def set_qrels(self, path):
        self.qrels = pd.read_csv(path)

    def analyez_docs(self, name_col):
        self.docs['analyzed_tokens'] = self.docs['content'].apply(analyez.analyze)
        self.docs[name_col] = self.docs['analyzed_tokens'].apply(
            lambda tokens: " ".join(tokens))

    def analyez_queries(self, name_col):
        self.queries['analyzed_tokens'] = self.queries['text'].apply(analyez.analyze)
        self.queries[name_col] = self.queries['analyzed_tokens'].apply(
            lambda tokens: " ".join(tokens))
