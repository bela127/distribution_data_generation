from typing import List, Tuple
import tensorflow as tf
from active_learning_ts.data_retrievement.data_source import DataSource as activeLearningDataSource


class DataSource(activeLearningDataSource):
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pass

    def query(self, actual_queries: List[tf.Tensor]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        queries = []
        results = []
        for x in actual_queries:
            q, r = self._query(x)
            queries.append(q)
            results.append(r)
        return queries, results

    def possible_queries(self):
        return None
