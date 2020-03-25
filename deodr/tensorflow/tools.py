"""Small tools for tensorflow."""

import numpy as np

import tensorflow as tf


def scipy_sparse_matrix_to_tensorflow(x):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)
