from unittest import TestCase
import tensorflow as tf
from active_learning_ts.pools.retrievement_strategies.nearest_neighbours_retreivement_strategy import \
    NearestNeighboursFindStrategy

from distribution_data_generation.data_sources.data_set_data_source import DataSetDataSource
from numpy import array


class TestDataSetDataSource(TestCase):

    def test_query(self):
        a = tf.constant([4.0, 4.0, 2.0, 2.0, 443.0, 3.0, 4.0, 344.0, 4.0])
        value = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        b = tf.constant([4.0, 4.0, 2.0, 2.0, 443.0, 3.0, 4.0, 344.0, 4.0])  # b is but a different object

        keys = array([a, tf.random.uniform(a.shape), tf.random.uniform(a.shape), tf.random.uniform(a.shape)])
        values = array([value, tf.random.uniform(a.shape), tf.random.uniform(a.shape), tf.random.uniform(a.shape)])

        source = DataSetDataSource(9, keys, values)

        source.post_init(retrievement_strategy=NearestNeighboursFindStrategy(1))

        expected = value
        x, actual = source.query([b])

        tf.assert_equal(expected, actual[0])
