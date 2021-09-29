from unittest import TestCase
from distribution_data_generation.data_sources.chaotic_data_source import ChaoticDataSource
import tensorflow as tf


class TestChaoticDataSource(TestCase):
    def test_query(self):
        s = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        source = ChaoticDataSource()
        a, actual = source.query(s)

        print(actual)
