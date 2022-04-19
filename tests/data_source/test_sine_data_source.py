import tensorflow as tf
from distribution_data_generation.data_sources.sine_data_source import SineDataSource


def test_query():
    cds = SineDataSource(4,2)

    print(cds.query([tf.constant([-0.9, 0, -0.5, 0.7])]))
