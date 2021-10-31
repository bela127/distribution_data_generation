import tensorflow as tf
from distribution_data_generation.data_sources.hourglass_data_source import HourglassDataSource


def test_query():
    cds = HourglassDataSource(2)

    print(cds.query([tf.constant([0.5, 0.1, 0.1, 0.1, 0.1])]))
