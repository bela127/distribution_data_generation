from typing import Protocol, List, Tuple
import tensorflow as tf
from os import environ
import multiprocessing.dummy as mp


class DataSource(Protocol):
    def __init__(self):
        pass

    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pass

    def query(self, actual_queries: List[tf.Tensor]) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        return [self._query(x) for x in actual_queries]
    # available_threads = 8
    #    try:
    #         available_threads = int(environ['query_threads'])
    #      except (ValueError, KeyError) as e:
    #           pass

    def possible_queries(self):
        return None
