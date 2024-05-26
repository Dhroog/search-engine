from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler


class TFIDF:
    def __init__(self, min_df, max_df):
        self.tfidf = TfidfVectorizer(stop_words='english',  # ngram_range=(2, 4),
                                     min_df=min_df,
                                     max_df=max_df)
        self.scaler = MaxAbsScaler()
        self.matrix = None
        self.queries_vec = None

    def init_tf_idf_matrix(self, docs, name_col):
        self.matrix = self.tfidf.fit_transform(docs[name_col])

    def scale_matrix(self):
        self.matrix = self.scaler.fit_transform(self.matrix)

    def init_tf_idf_vector(self, queries, name_col=None):
        if name_col is None:
            self.queries_vec = self.tfidf.transform([queries])
        else:
            self.queries_vec = self.tfidf.transform(queries[name_col])

    def scale_vector(self):
        self.queries_vec = self.scaler.transform(self.queries_vec)
