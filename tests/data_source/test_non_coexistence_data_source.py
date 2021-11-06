import tensorflow as tf
from distribution_data_generation.data_sources.non_coexistence_data_source import NonCoexistenceDataSource


def test_query():
    cds = NonCoexistenceDataSource(4)

    print(cds.query([tf.constant([-0.9, 0, -0.5, 0.7])]))
