from unittest import TestCase
from distribution_data_generation.data_sources.periodic_data_source import PeriodicDataSource
import tensorflow as tf


class TestQuadraticDataSource(TestCase):
    def test_query(self):
        c = tf.constant([243.0, 324.0, 234.0, 324.0, 234.0, 423.0, 243.0, 234.0, 23.0, 4.0, 324.0, 324.0, 2340., 3.0,
                         423.0, 324.0, 32.0, 4324.0, 234.0, 324.0])
        expected = tf.math.sin(c)
        source = PeriodicDataSource()
        a, actual = source.query(c)

        tf.assert_equal(expected, actual)

        assert True
