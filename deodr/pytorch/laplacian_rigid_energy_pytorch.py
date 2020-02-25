import numpy as np
import torch
from ..laplacian_rigid_energy import LaplacianRigidEnergy


def scipy_sparse_to_torch(sparse_matrix):
    coo = sparse_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices.astype(np.int32))
    v = torch.DoubleTensor(values)
    shape = coo.shape
    return torch.sparse.DoubleTensor(i, v, torch.Size(shape))


class LaplacianRigidEnergyPytorch(LaplacianRigidEnergy):
    def __init__(self, mesh, vertices, cregu):
        super().__init__(mesh, vertices, cregu)
        self.cT_torch = scipy_sparse_to_torch(self.cT)

    def eval(
        self, vertices, return_grad=True, return_hessian=True, refresh_rotations=True
    ):
        assert isinstance(vertices, torch.Tensor)
        if vertices.requires_grad:
            diff = (vertices - self.Vref).flatten()
            grad_vertices = self.cregu * (
                self.cT_torch.matmul(diff[:, None])
            ).reshape_as(vertices)
            energy = 0.5 * diff.dot(grad_vertices.flatten())
            return energy
        else:
            diff = (vertices - torch.tensor(self.Vref)).flatten()
            # gradV = self.cregu*(self.cT_torch.matmul(diff[:,None])).reshape_as(V)
            # 40x slower than scipy !
            grad_vertices = torch.tensor(
                self.cregu * (self.cT * (diff[:, None].numpy())).reshape(vertices.shape)
            )
            energy = 0.5 * diff.dot(grad_vertices.flatten())
            if not (return_grad):
                assert not (return_hessian)
                return energy
            if not (return_hessian):
                return energy, grad_vertices
            return energy, grad_vertices, self.approx_hessian
