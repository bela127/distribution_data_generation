from typing import Protocol
import tensorflow as tf


class DataSource(Protocol):
    def __init__(self):
        pass

    def query(self, actual_queries: tf.Tensor):
        pass

    def possible_queries(self):
        return None