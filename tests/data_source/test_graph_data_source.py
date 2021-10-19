from unittest import TestCase
from distribution_data_generation.data_sources.graph_data_source import GraphDataSource
import tensorflow as tf


class TestGraphDataSource(TestCase):
    def test_query(self):
        x = tf.constant([1, 2, 3, 4, 5, 6, 7])
        default_graph = GraphDataSource()
        a, result = default_graph.query([x])

        tf.assert_equal(x, result[0])

        def func(tensor):
            unstacked = tf.unstack(tensor)
            prev = unstacked[0]
            out = [prev]
            for i in unstacked[1:]:
                out.append(i + prev)
                prev = i
            return out

        add_graph = GraphDataSource(func)

        a, result = add_graph.query([x])

        expected = tf.constant([1, 3, 5, 7, 9, 11, 13])

        tf.assert_equal(expected, result[0])
