from typing import Protocol, List, Tuple
import tensorflow as tf


class DataSource(Protocol):
    def __init__(self):
        pass

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
