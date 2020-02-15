from scipy import sparse
import numpy as np
import tensorflow as tf
from ..triangulated_mesh import *
from .tools import scipy_sparse_matrix_to_tensorflow


class TriMeshAdjacenciesTensorflow(TriMeshAdjacencies):
    def __init__(self, faces):
        super().__init__(faces)
        self.faces_tf = tf.constant(faces)
        self.Vertices_Faces_tf = scipy_sparse_matrix_to_tensorflow(self.Vertices_Faces)

    def computeFaceNormals(self, vertices):
        tris = tf.gather(vertices, self.faces)
        u = tris[::, 1] - tris[::, 0]
        v = tris[::, 2] - tris[::, 0]
        if self.clockwise:
            n = -tf.linalg.cross(u, v)
        else:
            n = tf.linalg.cross(u, v)
        l = tf.sqrt(tf.reduce_sum(n ** 2, axis=1))
        return n / l[:, None]

    def computeVertexNormals(self, faceNormals):
        n = tf.sparse.sparse_dense_matmul(self.Vertices_Faces_tf, faceNormals)
        l = tf.sqrt(tf.reduce_sum(n ** 2, axis=1))
        return n / l[:, None]

    def edgeOnSilhouette(self, vertices2D):
        return super().edgeOnSilhouette(vertices2D.numpy())


class TriMeshTensorflow(TriMesh):
    def __init__(self, faces, vertices=None, clockwise=False):
        super().__init__(faces, vertices, clockwise)

    def computeAdjacencies(self):
        self.adjacencies = TriMeshAdjacenciesTensorflow(self.faces)


class ColoredTriMeshTensorflow(TriMeshTensorflow):
    def __init__(
        self,
        faces,
        vertices=None,
        clockwise=False,
        faces_uv=None,
        uv=None,
        texture=None,
        colors=None,
    ):
        super(ColoredTriMeshTensorflow, self).__init__(
            faces, vertices=vertices, clockwise=clockwise
        )
        self.faces_uv = faces_uv
        self.uv = uv
        self.texture = texture
        self.colors = colors
        self.textured = not (self.texture is None)

    def setVerticesColors(self, colors):
        self.verticesColors = colors
