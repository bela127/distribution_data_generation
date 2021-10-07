from typing import Tuple

from scipy.spatial.distance import cdist
import numpy as np
import tensorflow as tf
from distribution_data_generation.data_source import DataSource


def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


class GaussianDataSource(DataSource):
    """
    implementation of gaussian random function generation from
     https://peterroelants.github.io/posts/gaussian-process-tutorial/

     no further comments
    """
    # np array of np arrays with values at the given sample points
    functions = None

    lower_bound = None
    upper_bound = None
    sample_distance = None

    NUM_SAMPLES = 100

    def __init__(self, dimensions: int, lower_bound, upper_bound):
        super().__init__()
        number_of_functions = dimensions
        # Independent variable samples
        x = np.expand_dims(np.linspace(lower_bound, upper_bound, self.NUM_SAMPLES), 1)
        s = exponentiated_quadratic(x, x)  # Kernel of data points
        self.lower_bound = tf.constant(float(lower_bound))
        self.upper_bound = tf.constant(float(upper_bound))
        self.sample_distance = tf.constant(float((self.upper_bound - self.lower_bound) / (self.NUM_SAMPLES - 1)))

        # Draw samples from the prior at our data points.
        # Assume a mean of 0 for simplicity
        functions = np.random.multivariate_normal(
            mean=np.zeros(self.NUM_SAMPLES),
            cov=s,
            size=number_of_functions)
        self.functions = []
        for i in functions:
            self.functions.append([tf.constant(x) for x in i])

    def single_dim_query(self, function_index: int, sample_point: int):
        if sample_point <= self.lower_bound or sample_point >= self.upper_bound:
            return 0.0

        # interpolate between two nearest points
        sample_number = float(int((sample_point - self.lower_bound) / self.sample_distance))
        dist_to_lower_sample = sample_point - (sample_number * self.sample_distance + self.lower_bound)
        ratio_of_distance_to_upper_sample = dist_to_lower_sample / self.sample_distance

        temp = float(self.functions[function_index][int(sample_number)]) * (1 - ratio_of_distance_to_upper_sample)
        temp2 = float(self.functions[function_index][int(sample_number) + 1]) * ratio_of_distance_to_upper_sample

        return temp + temp2

    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        query_array = tf.unstack(tf.cast(actual_queries, dtype=tf.float32))
        out = []
        for i in range(0, len(query_array)):
            out.append(self.single_dim_query(i, query_array[i]))
        out = tf.stack(out)
        return actual_queries, out
