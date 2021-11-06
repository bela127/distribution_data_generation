import tensorflow as tf
from distribution_data_generation.data_sources.double_linear_data_source import DoubleLinearDataSource


def test_query():
    cds = DoubleLinearDataSource(4)

    print(cds.query([tf.constant([-0.9, 0, -0.5, 0.7])]))
