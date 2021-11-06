import tensorflow as tf
from distribution_data_generation.data_sources.linear_then_noise_data_source import LinearThenNoiseDataSource


def test_query():
    cds = LinearThenNoiseDataSource(4, dependency_dimension=4)

    print(cds.query([tf.constant([-0.9, 0, -0.5, 0.7])]))
