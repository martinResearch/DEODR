"""Tensorflow implementation of a triangulated mesh."""

import tensorflow as tf

from .tools import scipy_sparse_matrix_to_tensorflow
from ..triangulated_mesh import TriMesh, TriMeshAdjacencies, ColoredTriMesh


class TriMeshAdjacenciesTensorflow(TriMeshAdjacencies):
    """Class that stores adjacency matrices and methods that use this adjacencies using tensorflow sparse matrices.
    Unlike the TriMesh class there are no vertices stored in this class.
    """

    def __init__(self, faces, nb_vertices=None):
        super().__init__(faces=faces, nb_vertices=nb_vertices)
        self.faces_tf = tf.constant(faces)
        self._vertices_faces_tf = scipy_sparse_matrix_to_tensorflow(
            self._vertices_faces
        )

    def compute_face_normals(self, vertices):
        tris = tf.gather(vertices, self.faces)
        u = tris[::, 1] - tris[::, 0]
        v = tris[::, 2] - tris[::, 0]
        if self.clockwise:
            n = -tf.linalg.cross(u, v)
        else:
            n = tf.linalg.cross(u, v)
        norm = tf.sqrt(tf.reduce_sum(n ** 2, axis=1))
        return n / norm[:, None]

    def compute_vertex_normals(self, face_normals):
        n = tf.sparse.sparse_dense_matmul(self._vertices_faces_tf, face_normals)
        norm = tf.sqrt(tf.reduce_sum(n ** 2, axis=1))
        return n / norm[:, None]

    def edge_on_silhouette(self, vertices_2d):
        return super().edge_on_silhouette(vertices_2d.numpy())


class TriMeshTensorflow(TriMesh):
    """Tensorflow implementation of a triangulated mesh."""

    def __init__(self, faces, vertices=None, nb_vertices=None, clockwise=False):
        super().__init__(
            faces, vertices=vertices, nb_vertices=nb_vertices, clockwise=clockwise
        )

    def compute_adjacencies(self):
        self.adjacencies = TriMeshAdjacenciesTensorflow(
            self.faces, nb_vertices=self.nb_vertices
        )


class ColoredTriMeshTensorflow(TriMeshTensorflow, ColoredTriMesh):
    """Tensorflow implementation of a colored triangulated mesh."""

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
        self.vertices_colors = colors
