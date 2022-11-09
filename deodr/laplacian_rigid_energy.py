"""Implementation of an as-rigid-as-possible energy based on the difference of laplacian with a reference shape."""
from typing import Tuple
import numpy as np

import copy
import scipy
import scipy.sparse

from deodr.triangulated_mesh import TriMesh


class LaplacianRigidEnergy:
    """Class that implements an as-rigid-as-possible energy based on the difference of laplacian with a reference shape."""

    def __init__(self, mesh: TriMesh, vertices: np.ndarray, cregu: float):

        self.cT = scipy.sparse.kron(
            mesh.adjacencies.laplacian.T * mesh.adjacencies.laplacian,
            scipy.sparse.eye(3),
        ).tocsr()
        self.vertices_ref = copy.copy(vertices)
        self.mesh = mesh
        self.cregu = cregu
        self.approx_hessian = self.cregu * self.cT
        n_components, _ = scipy.sparse.csgraph.connected_components(
            csgraph=self.mesh.adjacencies.adjacency_vertices,
            directed=False,
            return_labels=True,
        )
        if n_components > 1:
            raise (
                BaseException(
                    "You have more than one connected component in your mesh."
                )
            )

    def evaluate(
        self,
        vertices: np.ndarray,
    ) -> Tuple[float, np.ndarray, scipy.sparse.csr_matrix]:

        diff = (vertices - self.vertices_ref).flatten()
        grad_vertices = self.cregu * (self.cT * diff).reshape((vertices.shape[0], 3))
        energy = 0.5 * diff.dot(grad_vertices.flatten())

        return energy, grad_vertices, self.approx_hessian
