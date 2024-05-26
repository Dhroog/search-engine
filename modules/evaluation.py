import pandas as pd
import numpy as np
from recommenders.evaluation import python_evaluation


# from recommenders.evaluation import python_evaluation
# from main import lotte_engine
class Evaluation:
    def __init__(self, core, k=10):
        self.k = k
        self.core = core
        self.prediction = None

    def init_queries_array(self):
        self.core.dataset.analyez_queries('analyzed_text')
        self.core.tf_idf.init_tf_idf_vector(self.core.dataset.queries, 'analyzed_text')
        self.core.tf_idf.scale_vector()

    def init_predection_array(self):
        distances, indices = self.core.nbrs.kneighbors(self.core.tf_idf.queries_vec)
        t1 = pd.melt(pd.DataFrame(distances).reset_index(), id_vars='index').rename(
            {'variable': 'col', 'index': 'row', 'value': 'score'}, axis=1)
        t2 = pd.melt(pd.DataFrame(indices).reset_index(), id_vars='index').rename(
            {'index': 'row', 'variable': 'col', 'value': 'doc_index'}, axis=1)
        self.prediction = t1.merge(t2,
                                   on=('row', 'col')).merge(self.core.dataset.queries['query_id'].reset_index(),
                                                            left_on='row', right_on='index').drop(
            ['index', 'row', 'col'],
            axis=1).merge(
            self.core.dataset.docs['doc_id'].reset_index(),
            left_on='doc_index', right_on='index').drop(['index',
                                                         'doc_index'], axis=1)
        self.prediction['score'] = 1 - self.prediction['score']

    def precision_at_k(self, query_id_col='query_id', doc_id_col='doc_id', score_col='score'):
        precision_at_k = python_evaluation.precision_at_k(
            self.core.dataset.qrels,
            self.prediction,
            query_id_col,
            doc_id_col,
            score_col,
            'top_k',
            k=self.k)
        return precision_at_k

    def recall_at_k(self, query_id_col='query_id', doc_id_col='doc_id', score_col='score'):
        recall_at_k = python_evaluation.recall_at_k(
            self.core.dataset.qrels,
            self.prediction,
            query_id_col,
            doc_id_col,
            score_col,
            'top_k',
            k=self.k)
        return recall_at_k

    def map_at_k(self, query_id_col='query_id', doc_id_col='doc_id', score_col='score'):
        map_score = python_evaluation.map_at_k(
            self.core.dataset.qrels,
            self.prediction,
            query_id_col,
            doc_id_col,
            score_col,
            'top_k',
            k=self.k)
        return map_score

    def mean_reciprocal_rank(self, query_id_col='query_id', doc_id_col='doc_id', score_col='score'):
        # Merge predictions with ground truth relevance
        merged = pd.merge(self.prediction, self.core.dataset.qrels, on=[query_id_col, doc_id_col],
                          how='left').fillna(0)
        merged = merged.sort_values(by=[query_id_col, score_col], ascending=[True, False])

        # Initialize reciprocal ranks
        reciprocal_ranks = []

        for query_id, group in merged.groupby(query_id_col):
            # Get top K results
            top_k = group.head(self.k)

            # Find the rank of the first relevant document
            for rank, (_, row) in enumerate(top_k.iterrows(), start=1):
                if row['relevance'] > 0:  # Assuming relevance > 0 indicates relevant document
                    reciprocal_ranks.append(1 / rank)
                    break
            else:
                reciprocal_ranks.append(0)

        # Calculate MRR
        mrr = np.mean(reciprocal_ranks)
        return mrr
