from distribution_data_generation.data_source import DataSource


class LinearDataSource(DataSource):
    def __init__(self):
        super().__init__()

    def query(self, actual_queries):
        result = actual_queries
        return result
