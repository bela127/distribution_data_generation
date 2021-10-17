from typing import Protocol, List, Tuple
import tensorflow as tf


class DataSource(Protocol):
    def __init__(self):
        pass

    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pass

    # TODO for convenience, change this from list[tuple]  to tuple[list]
    def query(self, actual_queries: List[tf.Tensor]) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        return [self._query(x) for x in actual_queries]

    def possible_queries(self):
        return None
