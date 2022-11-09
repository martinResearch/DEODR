# type: ignore
"""Pytorch implementation of a triangulated mesh."""

from typing import Callable, Optional
import numpy as np

import torch
from torch.sparse import DoubleTensor  # type: ignore

from ..triangulated_mesh import ColoredTriMesh, TriMeshAdjacencies


def print_grad(name: str) -> Callable[[torch.Tensor], None]:
    def hook(grad: torch.Tensor) -> None:
        print(f"grad {name} = {grad}")

    return hook


class TriMeshAdjacenciesPytorch(TriMeshAdjacencies):
    """Class that stores adjacency matrices and methods that use this adjacencies using pytorch sparse matrices.
    Unlike the TriMesh class there are no vertices stored in this class.
    """

    def __init__(self, faces: np.ndarray, clockwise: bool = False):
        super().__init__(faces, clockwise)
        self.faces_torch = torch.LongTensor(faces)
        i = self.faces_torch.flatten()
        j = torch.LongTensor(
            np.tile(np.arange(self.nb_faces)[:, None], [1, 3]).flatten()
        )
        self._vertices_faces_torch = DoubleTensor(
            torch.stack((i, j)),
            torch.ones((self.nb_faces, 3), dtype=torch.float64).flatten(),
            torch.Size((self.nb_vertices, self.nb_faces)),
        )

    def compute_face_normals(self, vertices: torch.Tensor) -> torch.Tensor:
        triangles = vertices[self.faces_torch, :]
        u = triangles[::, 1] - triangles[::, 0]
        v = triangles[::, 2] - triangles[::, 0]
        n = -torch.cross(u, v) if self.clockwise else torch.cross(u, v)
        l2 = (n**2).sum(dim=1)
        norm = l2.sqrt()
        return n / norm[:, None]

    def compute_vertex_normals(self, face_normals: torch.Tensor) -> torch.Tensor:
        n = self._vertices_faces_torch.mm(face_normals)
        l2 = (n**2).sum(dim=1)
        norm = l2.sqrt()
        return n / norm[:, None]

    def edge_on_silhouette(self, vertices_2d: torch.Tensor) -> np.ndarray:
        return super().edge_on_silhouette(vertices_2d.detach().numpy())


class ColoredTriMeshPytorch(ColoredTriMesh):
    """Pytorch implementation of colored a triangulated mesh."""

    def __init__(
        self,
        faces: np.ndarray,
        vertices: np.ndarray,
        clockwise: bool = False,
        faces_uv: Optional[np.ndarray] = None,
        uv: Optional[np.ndarray] = None,
        texture: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
    ):
        super(ColoredTriMeshPytorch, self).__init__(
            faces,
            vertices=vertices,
            clockwise=clockwise,
            faces_uv=faces_uv,
            uv=uv,
            texture=texture,
            colors=colors,
        )

    def compute_adjacencies(self) -> None:
        self._adjacencies = TriMeshAdjacenciesPytorch(self.faces)

    def set_vertices_colors(self, colors: torch.Tensor) -> None:
        self.vertices_colors = colors
