from distribution_data_generation.data_sources.linear_data_source import LinearDataSource
from unittest import TestCase
import tensorflow as tf


class TestLinearDataSource(TestCase):
    def test_query(self):
        c = tf.constant([243, 324, 234, 324, 234, 423, 243, 234, 23, 4, 324, 324, 234, 3, 423, 324, 32, 4324, 234, 324])
        expected = c
        a, source = LinearDataSource()

        tf.assert_equal(c, source.query(c))

        assert True
