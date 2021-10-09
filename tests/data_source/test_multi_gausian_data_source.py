from unittest import TestCase

from distribution_data_generation.data_sources.multi_gausian_data_source import MultiGausianDataSource
import tensorflow as tf


class TestMultiGausianDataSource(TestCase):

    def test_query(self):
        x1 = [tf.constant([-1.0]), tf.constant([-.9]), tf.constant([-.7]), tf.constant([-.4]), tf.constant([.0]),
              tf.constant([.1]), tf.constant([.4]), tf.constant([.7]), tf.constant([.9]), tf.constant([1.0])] * tf.constant(10.0)

        gdr = MultiGausianDataSource(in_dim=1, out_dim=1)
        a = gdr.query(x1)
        print([float(x[1]) for x in a])
