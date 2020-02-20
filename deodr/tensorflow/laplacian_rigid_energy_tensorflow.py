import tensorflow as tf
from ..laplacian_rigid_energy import LaplacianRigidEnergy
from .tools import scipy_sparse_matrix_to_tensorflow


class LaplacianRigidEnergyTensorflow(LaplacianRigidEnergy):
    def __init__(self, mesh, vertices, cregu):
        super().__init__(mesh, vertices, cregu)
        self.cT_tf = scipy_sparse_matrix_to_tensorflow(self.cT)

    def eval(self, V, return_grad=True, return_hessian=True):
        assert isinstance(V, tf.Tensor)
        diff = tf.reshape(V - tf.constant(self.Vref), [-1])
        gradV = tf.reshape(
            self.cregu * tf.sparse.sparse_dense_matmul(self.cT_tf, diff[:, None]),
            V.shape,
        )  # 5 times slower than scipy
        E = 0.5 * tf.reduce_sum(diff * tf.reshape(gradV, [-1]))
        if not (return_grad):
            assert not (return_hessian)
            return E
        if not (return_hessian):
            return E, gradV
        return E, gradV, self.approx_hessian
