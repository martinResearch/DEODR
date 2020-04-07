"""Implementation of an as-rigi-as-possible energy based on the difference of laplacian with a reference shape."""

import copy

import scipy
import scipy.sparse


class LaplacianRigidEnergy:
    """Class that implements an as-rigi-as-possible energy based on the difference of laplacian with a reference shape."""

    def __init__(self, mesh, vertices, cregu):
        self.cT = scipy.sparse.kron(
            mesh.adjacencies.Laplacian.T * mesh.adjacencies.Laplacian,
            scipy.sparse.eye(3),
        ).tocsr()
        self.vertices_ref = copy.copy(vertices)
        self.mesh = mesh
        self.cregu = cregu
        self.approx_hessian = self.cregu * self.cT
        n_components, labels = scipy.sparse.csgraph.connected_components(
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
        self, vertices, return_grad=True, return_hessian=True, refresh_rotations=True
    ):

        diff = (vertices - self.vertices_ref).flatten()
        grad_vertices = self.cregu * (self.cT * diff).reshape((vertices.shape[0], 3))
        energy = 0.5 * diff.dot(grad_vertices.flatten())
        if not (return_grad):
            assert not (return_hessian)
            return energy
        if not (return_hessian):
            return energy, grad_vertices

        return energy, grad_vertices, self.approx_hessian
