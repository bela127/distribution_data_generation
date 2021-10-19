from unittest import TestCase

import tensorflow as tf
from distribution_data_generation.data_sources.gausian_data_source import GaussianDataSource


class TestGausianDataSource(TestCase):
    def test_query(self):
        dims = 6
        lower_bound = -6.5
        upper_bound = 8.5

        source = GaussianDataSource(dims, lower_bound, upper_bound)

        a = tf.constant([-6.49, 0.72, 0.82, 0.92, 0.92, 8.4])

        x, out = source.query([a])

        print(out)
        print([float(x) for x in source.functions[5]])
        b = tf.constant([-10.0, -6.9, -6.5, 8.5, 8.9, 100.0])
        x, actual = source.query([b])

        expected = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        tf.assert_equal(actual[0], expected)
