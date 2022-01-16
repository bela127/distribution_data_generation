from unittest import TestCase
import tensorflow as tf
from active_learning_ts.pools.retrievement_strategies.nearest_neighbours_retreivement_strategy import \
    NearestNeighboursFindStrategy

from distribution_data_generation.data_sources.data_set_data_source import DataSetDataSource


class TestDataSetDataSource(TestCase):

    def test_query(self):
        a = tf.constant([4.0, 4.0, 2.0, 2.0, 443.0, 3.0, 4.0, 344.0, 4.0])
        value = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        b = tf.constant([4.0, 4.0, 2.0, 2.0, 443.0, 3.0, 4.0, 344.0, 4.0])  # b is but a different object

        keys = tf.convert_to_tensor(
            [a, tf.random.uniform(a.shape), tf.random.uniform(a.shape), tf.random.uniform(a.shape)])
        values = tf.convert_to_tensor(
            [value, tf.random.uniform(a.shape), tf.random.uniform(a.shape), tf.random.uniform(a.shape)])

        source = DataSetDataSource(keys, values)

        find = NearestNeighboursFindStrategy(1)
        source.post_init(retrievement_strategy=find)
        find.post_init(data_source=source)

        expected = value

        query = source.possible_queries().get_elements(tf.convert_to_tensor([b]))
        x, actual = source.query(query)

        tf.assert_equal(expected, actual[0])
