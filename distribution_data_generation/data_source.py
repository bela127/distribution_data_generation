from typing import List, Tuple
import tensorflow as tf
from active_learning_ts.data_retrievement.data_source import DataSource as activeLearningDataSource
from active_learning_ts.pool import Pool


class DataSource(activeLearningDataSource):

    def __init__(self) -> None:
        # TODO implement these in all subclasses
        self.point_shape = None
        self.value_shape = None

    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pass

    def query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        queries = []
        results = []
        for x in actual_queries:
            q, r = self._query(x)
            queries.append(q)
            results.append(r)
        return tf.convert_to_tensor(queries), tf.convert_to_tensor(results)

    def possible_queries(self) -> Pool:
        pass
