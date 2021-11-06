from distribution_data_generation.data_sources.linear_data_source import LinearDataSource
from unittest import TestCase
import tensorflow as tf


class TestLinearDataSource(TestCase):
    def test_query(self):
        c = tf.constant([243, 324, 234, 324, 234, 423, 243, 234, 23, 4, 324, 324, 234, 3, 423, 324, 32, 4324, 234, 324])
        expected = c
        source = LinearDataSource(c.shape[0])

        x, actual = source.query([c])
        tf.assert_equal(expected, actual[0])

        assert True
