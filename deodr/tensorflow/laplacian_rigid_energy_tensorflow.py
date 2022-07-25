"""Tensorflow implementation of an as-rigid-as-possible energy based on the difference of laplacian with a reference shape."""
from typing import Tuple

import numpy as np
import tensorflow as tf
from scipy.sparse import spmatrix

from .tools import scipy_sparse_matrix_to_tensorflow
from ..laplacian_rigid_energy import LaplacianRigidEnergy
from .triangulated_mesh_tensorflow import ColoredTriMeshTensorflow


class LaplacianRigidEnergyTensorflow(LaplacianRigidEnergy):
    """Tensorflow class that implements an as-rigid-as-possible energy based on the difference of laplacian with a reference shape."""

    def __init__(
        self, mesh: ColoredTriMeshTensorflow, vertices: np.ndarray, cregu: float
    ):
        super().__init__(mesh, vertices, cregu)
        self.cT_tf = scipy_sparse_matrix_to_tensorflow(self.cT)

    def evaluate(self, vertices: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, spmatrix]:
        assert isinstance(vertices, tf.Tensor)
        diff = tf.reshape(vertices - tf.constant(self.vertices_ref), [-1])
        grad_vertices = tf.reshape(
            self.cregu * tf.sparse.sparse_dense_matmul(self.cT_tf, diff[:, None]),
            vertices.shape,
        )  # 5 times slower than scipy
        energy = 0.5 * tf.reduce_sum(diff * tf.reshape(grad_vertices, [-1]))

        return energy, grad_vertices, self.approx_hessian
