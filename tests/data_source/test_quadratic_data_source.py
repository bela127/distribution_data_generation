from distribution_data_generation.data_sources.quadratic_data_source import QuadraticDataSource
from unittest import TestCase
import tensorflow as tf


class Test(TestCase):
    def test_query(self):
        s = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        quadratic_data_source = QuadraticDataSource()
        expected = s * s
        a, actual = quadratic_data_source.query([s])

        tf.debugging.assert_equal(
            expected, actual[0]
        )

        assert True
