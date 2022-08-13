"""Tensorflow implementation of a triangulated mesh."""
import numpy as np
import tensorflow as tf
from typing import Optional

from .tools import scipy_sparse_matrix_to_tensorflow
from ..triangulated_mesh import ColoredTriMesh, TriMeshAdjacencies


class TriMeshAdjacenciesTensorflow(TriMeshAdjacencies):
    """Class that stores adjacency matrices and methods that use this adjacencies using tensorflow sparse matrices.
    Unlike the TriMesh class there are no vertices stored in this class.
    """

    def __init__(self, faces: np.ndarray):
        super().__init__(faces)
        self.faces_tf = tf.constant(faces)
        self._vertices_faces_tf = scipy_sparse_matrix_to_tensorflow(
            self._vertices_faces
        )

    def compute_face_normals(self, vertices: tf.Tensor) -> tf.Tensor:
        tris = tf.gather(vertices, self.faces)
        u = tris[::, 1] - tris[::, 0]
        v = tris[::, 2] - tris[::, 0]
        n = -tf.linalg.cross(u, v) if self.clockwise else tf.linalg.cross(u, v)
        norm = tf.sqrt(tf.reduce_sum(n**2, axis=1))
        return n / norm[:, None]

    def compute_vertex_normals(self, face_normals: tf.Tensor) -> tf.Tensor:
        n = tf.sparse.sparse_dense_matmul(self._vertices_faces_tf, face_normals)
        norm = tf.sqrt(tf.reduce_sum(n**2, axis=1))
        return n / norm[:, None]

    def edge_on_silhouette(self, vertices_2d: tf.Tensor) -> np.ndarray:
        return super().edge_on_silhouette(vertices_2d.numpy())


class ColoredTriMeshTensorflow(ColoredTriMesh):
    """Tensorflow implementation of a colored triangulated mesh."""

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
        super(ColoredTriMeshTensorflow, self).__init__(
            faces,
            vertices=vertices,
            clockwise=clockwise,
            faces_uv=faces_uv,
            uv=uv,
            texture=texture,
            colors=colors,
        )

    def compute_adjacencies(self) -> None:
        self._adjacencies = TriMeshAdjacenciesTensorflow(self.faces)

    def set_vertices_colors(self, colors: tf.Tensor) -> None:
        self.vertices_colors = colors
