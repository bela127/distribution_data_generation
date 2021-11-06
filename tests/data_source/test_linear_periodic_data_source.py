import tensorflow as tf
from distribution_data_generation.data_sources.linear_periodic_data_source import LinearPeriodicDataSource


def test_query():
    cds = LinearPeriodicDataSource(4, 0.9)

    print(cds.query([tf.constant([0.9, 0, -0.5, 0.7])]))
