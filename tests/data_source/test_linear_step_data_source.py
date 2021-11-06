import tensorflow as tf
from distribution_data_generation.data_sources.linear_step_data_source import LinearStepDataSource


def test_query():
    cds = LinearStepDataSource(4, 0.9)

    print(cds.query([tf.constant([-0.9, 0, -0.5, 0.7])]))
