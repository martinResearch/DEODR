# type: ignore
"""Pytorch implementation of an as-rigid-as-possible energy based on the difference of laplacian with a reference shape."""

from typing import Tuple
import numpy as np
from scipy.sparse import spmatrix

import torch
from torch.sparse import DoubleTensor  # type: ignore

from ..laplacian_rigid_energy import LaplacianRigidEnergy
from .triangulated_mesh_pytorch import ColoredTriMeshPytorch


def scipy_sparse_to_torch(sparse_matrix: spmatrix) -> DoubleTensor:
    coo = sparse_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices.astype(np.int32))
    v = torch.DoubleTensor(values)
    shape = coo.shape
    return DoubleTensor(i, v, torch.Size(shape))


class LaplacianRigidEnergyPytorch:
    """Pytorch class that implements an as-rigid-as-possible energy based on the difference of laplacian with a reference shape."""

    def __init__(self, mesh: ColoredTriMeshPytorch, vertices: np.ndarray, cregu: float):
        self.numpy_imp = LaplacianRigidEnergy(mesh, vertices, cregu)
        self.cT_torch = scipy_sparse_to_torch(self.numpy_imp.cT)
        self.approx_hessian = scipy_sparse_to_torch(self.numpy_imp.approx_hessian)

    def evaluate(
        self,
        vertices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert isinstance(vertices, torch.Tensor)
        if vertices.requires_grad:
            diff = (vertices - self.numpy_imp.vertices_ref).flatten()
            grad_vertices = self.numpy_imp.cregu * (
                self.cT_torch.matmul(diff[:, None])
            ).reshape_as(vertices)
        else:
            diff = (vertices - torch.tensor(self.numpy_imp.vertices_ref)).flatten()
            # gradV = self.cregu*(self.cT_torch.matmul(diff[:,None])).reshape_as(V)
            # 40x slower than scipy !
            grad_vertices = torch.tensor(
                self.numpy_imp.cregu
                * (self.numpy_imp.cT * (diff[:, None].numpy())).reshape(vertices.shape)
            )

        energy = 0.5 * diff.dot(grad_vertices.flatten())
        return energy, grad_vertices, self.numpy_imp.approx_hessian
