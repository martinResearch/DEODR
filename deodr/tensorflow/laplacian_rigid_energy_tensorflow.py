"""Tensorflow implementation of an as-rigi-as-possible energy based on the difference of laplacian with a reference shape."""

import tensorflow as tf

from .tools import scipy_sparse_matrix_to_tensorflow
from ..laplacian_rigid_energy import LaplacianRigidEnergy


class LaplacianRigidEnergyTensorflow(LaplacianRigidEnergy):
    """Tensorflow class that implements an as-rigi-as-possible energy based on the difference of laplacian with a reference shape."""

    def __init__(self, mesh, vertices, cregu):
        super().__init__(mesh, vertices, cregu)
        self.cT_tf = scipy_sparse_matrix_to_tensorflow(self.cT)

    def evaluate(self, vertices, return_grad=True, return_hessian=True):
        assert isinstance(vertices, tf.Tensor)
        diff = tf.reshape(vertices - tf.constant(self.vertices_ref), [-1])
        grad_vertices = tf.reshape(
            self.cregu * tf.sparse.sparse_dense_matmul(self.cT_tf, diff[:, None]),
            vertices.shape,
        )  # 5 times slower than scipy
        energy = 0.5 * tf.reduce_sum(diff * tf.reshape(grad_vertices, [-1]))
        if not (return_grad):
            assert not (return_hessian)
            return energy
        if not (return_hessian):
            return energy, grad_vertices
        return energy, grad_vertices, self.approx_hessian
