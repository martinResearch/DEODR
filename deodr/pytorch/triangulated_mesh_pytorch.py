"""Pytorch implementation of a triangulated mesh."""


import numpy as np

import torch

from ..triangulated_mesh import TriMesh, TriMeshAdjacencies, ColoredTriMesh


def print_grad(name):
    def hook(grad):
        print(f"grad {name} = {grad}")

    return hook


class TriMeshAdjacenciesPytorch(TriMeshAdjacencies):
    """Class that stores adjacency matrices and methods that use this adjacencies using pytorch sparse matrices.
    Unlike the TriMesh class there are no vertices stored in this class.
    """

    def __init__(self, faces, clockwise=False, nb_vertices=None):
        super().__init__(faces=faces, clockwise=clockwise, nb_vertices=nb_vertices)
        self.faces_torch = torch.LongTensor(faces)
        i = self.faces_torch.flatten()
        j = torch.LongTensor(
            np.tile(np.arange(self.nb_faces)[:, None], [1, 3]).flatten()
        )
        self._vertices_faces_torch = torch.sparse.DoubleTensor(
            torch.stack((i, j)),
            torch.ones((self.nb_faces, 3), dtype=torch.float64).flatten(),
            torch.Size((self.nb_vertices, self.nb_faces)),
        )

    def compute_face_normals(self, vertices):
        triangles = vertices[self.faces_torch, :]
        u = triangles[::, 1] - triangles[::, 0]
        v = triangles[::, 2] - triangles[::, 0]
        if self.clockwise:
            n = -torch.cross(u, v)
        else:
            n = torch.cross(u, v)
        l2 = (n ** 2).sum(dim=1)
        norm = l2.sqrt()
        nn = n / norm[:, None]
        return nn

    def compute_face_areas(self, vertices):
        triangles = vertices[self.faces_torch, :]
        u = triangles[::, 1] - triangles[::, 0]
        v = triangles[::, 2] - triangles[::, 0]
        n = torch.cross(u, v)
        l2 = (n ** 2).sum(dim=1)
        areas = l2.sqrt()
        return areas

    def compute_vertex_normals(self, face_normals):
        n = self._vertices_faces_torch.mm(face_normals)
        l2 = (n ** 2).sum(dim=1)
        norm = l2.sqrt()
        return n / norm[:, None]

    def edge_on_silhouette(self, vertices_2d):
        return super().edge_on_silhouette(vertices_2d.detach().numpy())


class TriMeshPytorch(TriMesh):
    """Pytorch implementation of a triangulated mesh."""

    def __init__(
        self,
        faces,
        vertices=None,
        nb_vertices=None,
        clockwise=False,
        compute_adjacencies=True,
    ):
        TriMesh.__init__(
            self,
            faces,
            vertices=vertices,
            nb_vertices=nb_vertices,
            clockwise=clockwise,
            compute_adjacencies=compute_adjacencies,
        )

    def compute_adjacencies(self):
        self.adjacencies = TriMeshAdjacenciesPytorch(
            self.faces, nb_vertices=self.nb_vertices
        )

    def _data_as_numpy_array(self, data):
        return data.detach().numpy()


class ColoredTriMeshPytorch(TriMeshPytorch, ColoredTriMesh):
    """Pytorch implementation of colored a triangulated mesh."""

    def __init__(
        self,
        faces,
        vertices=None,
        nb_vertices=None,
        clockwise=False,
        faces_uv=None,
        uv=None,
        texture=None,
        colors=None,
        nb_colors=None,
        compute_adjacencies=True,
    ):
        ColoredTriMesh.__init__(
            self,
            faces,
            vertices=vertices,
            nb_vertices=nb_vertices,
            clockwise=clockwise,
            faces_uv=faces_uv,
            uv=uv,
            texture=texture,
            colors=colors,
            nb_colors=nb_colors,
            compute_adjacencies=compute_adjacencies,
        )


    def set_vertices_colors(self, colors):
        assert isinstance(colors, torch.Tensor)
        self.vertices_colors = colors
