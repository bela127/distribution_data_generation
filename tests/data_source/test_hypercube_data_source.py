import tensorflow as tf
from distribution_data_generation.data_sources.hypercube_data_source import HypercubeDataSource


def test_query():
    cds = HypercubeDataSource(4,1)

    print(cds._query(tf.constant([0.9, 0, 0.5, 0.7])))
