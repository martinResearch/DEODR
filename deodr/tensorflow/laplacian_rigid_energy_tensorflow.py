import tensorflow as tf
from ..laplacian_rigid_energy import LaplacianRigidEnergy
from .tools import scipy_sparse_matrix_to_tensorflow


class LaplacianRigidEnergyTensorflow(LaplacianRigidEnergy):
    def __init__(self, mesh, vertices, cregu):
        super().__init__(mesh, vertices, cregu)
        self.cT_tf = scipy_sparse_matrix_to_tensorflow(self.cT)

    def eval(self, vertices, return_grad=True, return_hessian=True):
        assert isinstance(vertices, tf.Tensor)
        diff = tf.reshape(vertices - tf.constant(self.Vref), [-1])
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
