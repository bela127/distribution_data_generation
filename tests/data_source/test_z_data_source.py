import tensorflow as tf
from distribution_data_generation.data_sources.z_data_source import ZDataSource


def test_query():
    cds = ZDataSource(4)

    print(cds.query([tf.constant([0.9, 0.1, 0.5, 0.7])]))
