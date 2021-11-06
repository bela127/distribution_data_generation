import tensorflow as tf
from distribution_data_generation.data_sources.power_data_source import PowerDataSource


def test_query():
    cds = PowerDataSource(4, 1.0/2.0)

    print(cds.query([tf.constant([-0.9, 0, -0.5, 0.7])]))
