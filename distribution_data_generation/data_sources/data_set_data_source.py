from distribution_data_generation.data_source import DataSource
import tensorflow as tf
from numpy import array


class DataSetDataSource(DataSource):
    """
    This solution sucks. tf.dataset isn't meant for this, neither is tf..hashtable. This can be changed in the future,
    i think there are tf based k-d-tree implementations one could use here
    """
    data_points = None
    data_values = None

    def __init__(self, data_points: array, data_values: array):
        super().__init__()
        self.data_points = data_points
        self.data_values = data_values

    def _query(self, actual_queries: tf.Tensor):
        place = 0
        for t in self.data_points.tolist():
            if tf.reduce_all(tf.math.equal(t, actual_queries)):
                break
            place += 1
        return actual_queries, self.data_values[place]
