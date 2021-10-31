import tensorflow as tf
from distribution_data_generation.data_sources.cubic_data_source import CubicDataSource


def test_query():
    cds = CubicDataSource()

    print(cds.query([tf.constant([-0.9, 0, -0.5, 0.7])]))
