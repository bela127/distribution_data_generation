import tensorflow as tf
from distribution_data_generation.data_sources.linear_then_dummy_data_source import LinearThenDummyDataSource


def test_query():
    cds = LinearThenDummyDataSource(4, dependency_dimension=4)

    print(cds.query([tf.constant([-0.9, 0, -0.5, 0.7])]))
