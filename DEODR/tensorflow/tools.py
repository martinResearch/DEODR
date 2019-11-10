from scipy import sparse
import numpy as np
import tensorflow as tf


def scipy_sparse_matrix_to_tensorflow(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)
