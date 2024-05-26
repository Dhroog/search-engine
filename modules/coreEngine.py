from modules.dataSet import DataSet
from modules import analyez
from modules.tfIdf import TFIDF
from sklearn.neighbors import NearestNeighbors


class CoreEngine:
    def __init__(self, min_df, max_df, path_docs, path_queries, path_grels, n_neighbors=20, metric='cosine'):
        self.dataset = DataSet(path_docs, path_queries, path_grels)
        self.dataset.analyez_docs('analyzed_text')
        self.tf_idf = TFIDF(min_df, max_df)
        self.tf_idf.init_tf_idf_matrix(self.dataset.docs, 'analyzed_text')
        self.tf_idf.scale_matrix()
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(self.tf_idf.matrix)

    def search(self, query):
        query_analyzed = analyez.analyze(query)
        query_text = " ".join(query_analyzed)
        self.tf_idf.init_tf_idf_vector(query_text)
        self.tf_idf.scale_vector()
        return self.matching()

    def matching(self):
        distances, indices = self.nbrs.kneighbors(self.tf_idf.queries_vec)
        matching_docs = self.dataset.docs.iloc[indices[0]]
        return matching_docs[['doc_id', 'content']].to_dict(orient="records")
