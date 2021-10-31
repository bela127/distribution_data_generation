import tensorflow as tf
from distribution_data_generation.data_sources.cross_data_source import CrossDataSource


def test_query():
    cds = CrossDataSource()

    print(cds.query([tf.constant([0.5, 0.1, 0.1, 0.1, 0.1])]))
