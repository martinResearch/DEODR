"""Tensorflow implementation of an as-rigid-as-possible energy based on the difference of laplacian with a reference shape."""
from typing import Tuple

import numpy as np
import tensorflow as tf

from .tools import scipy_sparse_matrix_to_tensorflow
from ..laplacian_rigid_energy import LaplacianRigidEnergy
from .triangulated_mesh_tensorflow import ColoredTriMeshTensorflow


class LaplacianRigidEnergyTensorflow:
    """Tensorflow class that implements an as-rigid-as-possible energy based on the difference of laplacian with a reference shape."""

    def __init__(
        self, mesh: ColoredTriMeshTensorflow, vertices: np.ndarray, cregu: float
    ):
        self.numpy_imp = LaplacianRigidEnergy(mesh, vertices, cregu)
        self.cT_tf = scipy_sparse_matrix_to_tensorflow(self.numpy_imp.cT)
        self.approx_hessian = scipy_sparse_matrix_to_tensorflow(
            self.numpy_imp.approx_hessian
        )

    def evaluate(
        self, vertices: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.SparseTensor]:
        assert isinstance(vertices, tf.Tensor)
        diff = tf.reshape(vertices - tf.constant(self.numpy_imp.vertices_ref), [-1])
        grad_vertices = tf.reshape(
            self.numpy_imp.cregu
            * tf.sparse.sparse_dense_matmul(self.cT_tf, diff[:, None]),
            vertices.shape,
        )  # 5 times slower than scipy
        energy = 0.5 * tf.reduce_sum(diff * tf.reshape(grad_vertices, [-1]))

        return energy, grad_vertices, self.approx_hessian
