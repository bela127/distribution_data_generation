from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class LinearDataSource(DataSource):
    def __init__(self, dim: int):
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim)
        self.point_shape = (dim,)
        self.value_shape = (dim,)

    def _query(self, actual_queries):
        result = actual_queries
        return actual_queries, result

    def possible_queries(self):
        return self.pool
