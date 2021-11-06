import tensorflow as tf
from distribution_data_generation.data_sources.star_data_source import StarDataSource


def test_query():
    cds = StarDataSource(4)

    print(cds.query([tf.constant([0.9, 0, .5, 0.7])]))
