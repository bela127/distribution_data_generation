from unittest import TestCase
import tensorflow as tf
from distribution_data_generation.data_sources.random_data_source import RandomDataSource


class TestRandomDataSource(TestCase):
    def test_query(self):
        c = tf.constant([45, 45, 4543, 34, 43, 54, 54, 43, 25, 34, 6545, 34, 324, 23, 55])

        source = RandomDataSource()

        a, b = source.query([c])
        print(b[0])

        assert True
